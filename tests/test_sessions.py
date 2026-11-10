import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import uuid
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"robot"))
from storage import SessionStore, component
from record_view import build_episode

class Collector:
    def __init__(self):self.starts=[];self.stops=[];self.fail_stop=False
    def start(self,sid,raw):
        self.starts.append(sid)
        (raw/"record").mkdir()
        (raw/"record/-remote-vr_data.pbdat").write_bytes(b"vr-raw")
        return {"uuid":sid,"raw_directory":str(raw)}
    def stop(self,sid,receipt):
        if self.fail_stop:raise TimeoutError("missing stop receipt")
        self.stops.append(sid);return {"confirmed":True}
    def readiness(self):return {"ready":True}

class SessionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.collector=Collector()
        self.store=SessionStore(self.tmp.name,self.collector,build_episode)
        self.sid=uuid.uuid4().hex;self.seq=0;self.event_seq=0
        self.store.create(self.sid,"开门","洗衣机",{"host":"113"})
        self.store.start(self.sid)
        self.add("metadata",{"task_id":"开门","config":{},"policy":{}})
        self.events([{"kind":"created","elapsed_s":0}])
    def add(self,kind,data,rid=None):
        self.seq+=1
        record={"schema":"action-chunk-events/1","episode_id":self.sid,
            "record_id":rid or uuid.uuid4().hex,"sequence":self.seq,
            "data_dir":"/data/yfmo","recorded_114":{"wall_ns":"1700000000000000000",
            "monotonic_ns":str(1000000000+self.seq*100000000),"host":"113","boot_id":"boot"},
            "kind":kind,"data":data}
        self.store.put_record(record);return record
    def events(self,items):
        events=[]
        for value in items:
            self.event_seq+=1
            events.append({"seq":self.event_seq,"wall_ns":str(1700000000000000000+round(value["elapsed_s"]*1e9)),
                "monotonic_ns":str(1000000000+round(value["elapsed_s"]*1e9)),**value})
        self.add("events",{"events":events})
    def three_rounds(self):
        self.events([{"kind":"model_started","elapsed_s":0.1},
            {"kind":"human_round_started","round":1,"elapsed_s":1},
            {"kind":"model_resumed","elapsed_s":2},
            {"kind":"human_round_started","round":2,"elapsed_s":3},
            {"kind":"model_resumed","elapsed_s":4},
            {"kind":"human_round_started","round":3,"elapsed_s":5},
            {"kind":"model_resumed","elapsed_s":6}])
    def finish(self):
        result={"success":True,"duration_s":7,"interventions":3,"reasons":[],"note":""}
        self.add("result",{"result":result});self.add("session_end",{"result":result})
        self.store.stop(self.sid)
    def annotate(self,round_id,reasons=None,note=""):
        revision=self.store.manifest(self.sid)["revision"]
        return self.store.annotate(self.sid,round_id,reasons or [],note,revision)
    def test_classified_directory_and_single_raw_copy(self):
        path=self.store.directory(self.sid)
        self.assertTrue(path.parent.parent.parent.name.startswith("洗衣机_"))
        self.assertTrue(path.parent.parent.name.startswith("开门_"))
        self.assertEqual((path/"raw/record/-remote-vr_data.pbdat").read_bytes(),b"vr-raw")
        self.assertFalse((path/"vr").exists())
    def test_component_traversal_and_aliases(self):
        for value in ["..","../outside","a/b","a\\b","\0"]:
            with self.assertRaises(ValueError):component(value)
        self.assertNotEqual(component("a:b"),component("a?b"))
    def test_same_record_is_idempotent_conflicts_are_rejected(self):
        r=self.add("chunk",{"chunk_id":uuid.uuid4().hex,"actions":[[0]*16]})
        self.assertTrue(self.store.put_record(r)["duplicate"])
        changed=copy.deepcopy(r);changed["data"]["actions"]=[[1]*16]
        with self.assertRaises(ValueError):self.store.put_record(changed)
    def test_three_rounds_require_three_annotations(self):
        self.three_rounds();self.finish()
        d=self.store.detail(self.sid);self.assertEqual(d["missing_annotations"],[1,2,3])
        with self.assertRaises(ValueError):self.store.finalize(self.sid,d["session_revision"])
        self.annotate(1,["grasp_failed"],"夹爪偏高")
        self.annotate(2,note="门把手处停滞")
        d=self.annotate(3,["wrong_pose"],"位置纠正")
        self.assertTrue(d["can_finalize"])
        final=self.store.finalize(self.sid,d["session_revision"])
        self.assertEqual(final["archive_state"],"complete")
        self.assertEqual(len(self.store.detail(self.sid)["rounds"]),3)
    def test_other_requires_description_but_draft_is_retained(self):
        self.three_rounds();d=self.annotate(1,["other"])
        self.assertIn(1,d["missing_annotations"])
        d=self.annotate(1,["other"],"抓住了门边")
        self.assertNotIn(1,d["missing_annotations"])
    def test_invalid_or_unknown_round_rejected(self):
        self.three_rounds()
        with self.assertRaises(ValueError):self.annotate(4,["wrong_pose"])
        with self.assertRaises(ValueError):self.annotate(1,["not_a_reason"])
    def test_capture_must_stop_before_finalize(self):
        self.add("result",{"result":{"success":True}})
        d=self.store.detail(self.sid);self.assertFalse(d["can_finalize"])
        with self.assertRaises(ValueError):self.store.finalize(self.sid,d["session_revision"])
    def test_missing_overall_result_blocks_finalize(self):
        self.store.stop(self.sid);d=self.store.detail(self.sid)
        self.assertFalse(d["can_finalize"])
    def test_stop_confirmation_failure_is_not_completed(self):
        self.collector.fail_stop=True
        with self.assertRaises(TimeoutError):self.store.stop(self.sid)
        self.assertEqual(self.store.manifest(self.sid)["collection_state"],"stop_failed")
        self.collector.fail_stop=False;self.store.stop(self.sid)
        self.assertEqual(self.store.manifest(self.sid)["collection_state"],"stopped")
    def test_stop_idempotent_and_restart_preserves_annotations(self):
        self.three_rounds();self.finish();self.annotate(1,note="第一次")
        self.store.stop(self.sid);self.assertEqual(len(self.collector.stops),1)
        restarted=SessionStore(self.tmp.name,self.collector,build_episode)
        self.assertEqual(restarted.detail(self.sid)["rounds"][0]["annotation"]["note"],"第一次")
    def test_concurrent_edits_require_current_revision(self):
        self.three_rounds();old=self.store.manifest(self.sid)["revision"]
        self.store.annotate(self.sid,1,["grasp_failed"],"",old)
        with self.assertRaises(ValueError):self.store.annotate(self.sid,2,[],"second",old)
    def test_active_recording_blocks_another_session(self):
        with self.assertRaises(ValueError):self.store.create(uuid.uuid4().hex,"关门","洗衣机")
        self.store.stop(self.sid)
        other=uuid.uuid4().hex;self.store.create(other,"关门","洗衣机")
        self.assertNotEqual(self.store.directory(other).parent.parent,self.store.directory(self.sid).parent.parent)
    def test_finalized_record_is_read_only(self):
        self.finish();d=self.store.detail(self.sid);self.store.finalize(self.sid,d["session_revision"])
        with self.assertRaises(ValueError):self.add("events",{"events":[]})
    def test_invalid_numeric_payload_not_saved(self):
        with self.assertRaises(ValueError):self.add("chunk",{"actions":[float("nan")]})
if __name__=="__main__":unittest.main()
