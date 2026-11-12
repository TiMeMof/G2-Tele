import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'host'),str(ROOT/'robot'),str(ROOT/'tests')]
from episode_control import Episode,Conflict
from takeover_runtime import Runtime
from chunk_log import Uploader,Recorder,encode
from storage import SessionStore
from record_view import build_episode
from test_sessions import Collector


class Bridge:
    def __init__(self,store):self.store=store;self.fail_annotation=False
    def anchor(self):return {}
    def request(self,method,path,body=None):
        parts=path.strip('/').split('/')
        if path=='/api/sessions':
            self.store.create(body['episode_id'],body['task_id'],body['category'],body.get('policy'))
            return {'success':True,'session':self.store.start(body['episode_id'])}
        if path.endswith('/stop'):return {'success':True,'session':self.store.stop(parts[2],body.get('expected_sequence'))}
        if path=='/api/records':return {'success':True,**self.store.put_record(body['record'])}
        if len(parts)==3:return {'success':True,'episode':self.store.detail(parts[2])}
        if parts[-1]=='annotations':
            if self.fail_annotation:self.fail_annotation=False;raise OSError('connection lost')
            value=self.store.annotate(parts[2],body['round'],body['reasons'],body['note'],body['revision'])
        else:value=self.store.finalize(parts[2],body['revision'])
        return {'success':True,'episode':value}


class InlineReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name);self.collector=Collector()
        self.store=SessionStore(root/'robot',self.collector,build_episode)
        self.bridge=Bridge(self.store);self.u=Uploader(root/'spool',self.bridge)
        self.now=0.;self.ep=Episode('开门',{'max_interventions':None,'data_dir':'/data/yfmo'}, {},lambda:self.now)
        self.runtime=Runtime(lambda:(200,{}),readiness_check=lambda:{'ready':True})
        self.runtime.episode=self.ep;self.ep.recording=Recorder(self.ep,self.u)
        self.ep.recording.write('metadata',{'task_id':'开门','config':{}})
        self.feed(False);token=self.ep.begin_initialization();self.ep.initialized(token,True)
        for _ in range(2):
            self.feed(True);self.now+=.01;self.feed(False)
            for _ in range(4):self.now+=.1;self.feed(False)
        self.ep.command('pause',self.ep.id,self.ep.revision,'manual-stop')
    def feed(self,active):
        self.ep.observe_vr({'fresh':True,'takeover':active,'age_ms':0,'instance_id':'test','connection':{'connected':True}})
    def begin(self):
        self.runtime.command({'command':'review','episodeId':self.ep.id,'revision':self.ep.revision,'requestId':'open-review'})
        self.ep.recording.write('session_end',{'phase':'awaiting_result','result':None})
        self.ep.recording.stop_collection(refresh_barrier=True)
    def payload(self):return {'episodeId':self.ep.id,'round_annotations':[
        {'round':1,'reasons':['grasp_failed'],'note':'夹爪偏高'},
        {'round':2,'reasons':[],'note':'门把手处停滞'}],'reasons':[],'note':''}
    def test_open_stops_raw_but_does_not_save_result_or_annotations(self):
        self.begin();d=self.store.detail(self.ep.id)
        self.assertEqual(d['collection_state'],'stopped')
        self.assertEqual(d['archive_state'],'draft');self.assertIsNone(d['result'])
        self.assertEqual(d['missing_annotations'],[1,2]);self.assertTrue(self.runtime.active())
        self.assertTrue(self.runtime.review_ready())
    def test_save_all_rounds_archives_once_and_excludes_label_time(self):
        self.begin();duration=self.ep.review_started_s;self.now+=100
        value=self.runtime.save_review(self.payload())
        self.assertEqual(value['review']['archive_state'],'complete')
        self.assertEqual(value['review']['duration_s'],duration)
        self.assertFalse(value['review']['result']['success'])
        self.assertEqual(value['review']['rounds'][1]['annotation']['note'],'门把手处停滞')
        self.assertFalse(self.runtime.active());self.assertEqual(len(self.collector.stops),1)
        count=len(self.store.records(self.ep.id));self.runtime.save_review(self.payload())
        self.assertEqual(len(self.store.records(self.ep.id)),count)
    def test_missing_annotation_does_not_submit_failure(self):
        self.begin();p=self.payload();p['round_annotations'][1]['note']=''
        with self.assertRaises(ValueError):self.runtime.save_review(p)
        self.assertIsNone(self.ep.result);self.assertIsNone(self.store.manifest(self.ep.id)['result'])
    def test_partial_network_failure_keeps_pending_review_and_can_retry(self):
        self.begin();self.bridge.fail_annotation=True
        with self.assertRaises(OSError):self.runtime.save_review(self.payload())
        self.assertTrue(self.runtime.active());self.assertEqual(self.store.manifest(self.ep.id)['archive_state'],'draft')
        self.runtime.command({'command':'review','episodeId':self.ep.id,'revision':self.ep.revision,'requestId':'reopen-after-refresh'})
        self.assertEqual(len(self.ep.snapshot()['review_submission']['round_annotations']),2)
        result=self.runtime.save_review(self.payload())
        self.assertEqual(result['review']['archive_state'],'complete');self.assertFalse(self.runtime.active())
    def test_review_never_resumes_on_release(self):
        self.begin();self.now+=1;self.feed(False)
        self.assertEqual(self.ep.phase,'awaiting_result');self.assertFalse(self.ep.snapshot()['can_resume'])
        self.assertEqual(self.ep.snapshot()['elapsed_s'],self.ep.review_started_s)


if __name__=='__main__':unittest.main()
