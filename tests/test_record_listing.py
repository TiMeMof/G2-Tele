import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import uuid
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'host'))
import record_browser
from record_browser import RecordBrowser
from remote_records import RemoteRecordBrowser


class Offline:
    def request(self,*args,**kwargs):raise OSError('robot offline')


class RecordListingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
    def add(self,index):
        sid=uuid.uuid4().hex;rid=uuid.uuid4().hex;folder=self.root/sid;folder.mkdir()
        value={'schema':'action-chunk-events/1','episode_id':sid,'record_id':rid,'sequence':1,
            'kind':'metadata','data':{'task_id':'任务'+str(index)},
            'recorded_114':{'wall_ns':str(1700000000000000000+index),'monotonic_ns':'100','boot_id':'one'}}
        path=folder/(rid+'.json');path.write_text(json.dumps(value),encoding='utf-8');return sid,path,value
    def test_listing_cache_survives_more_than_eight_sessions_and_detects_edits(self):
        rows=[self.add(i) for i in range(12)];browser=RecordBrowser(self.root)
        self.assertEqual(browser.listing({})['total'],12)
        with mock.patch.object(record_browser.json,'loads',wraps=json.loads) as parse:
            self.assertEqual(browser.listing({})['total'],12)
            self.assertEqual(parse.call_count,0)
        sid,path,value=rows[-1];value['data']['task_id']='内容已修改'
        path.write_text(json.dumps(value),encoding='utf-8')
        result=browser.listing({'task':['内容已修改']})
        self.assertEqual([r['episode_id'] for r in result['records']],[sid])
    def test_legacy_pagination_does_not_drop_records_after_first_hundred(self):
        for i in range(105):self.add(i)
        browser=RemoteRecordBrowser(self.root,Offline())
        result=browser.listing({'offset':['100'],'limit':['50']})
        self.assertEqual(result['total'],105);self.assertEqual(len(result['records']),5)
        self.assertTrue(result['errors']);self.assertTrue(all(r['legacy'] for r in result['records']))


if __name__=='__main__':unittest.main()
