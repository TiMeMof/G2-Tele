import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import uuid
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"host"))
from chunk_log import Uploader, Recorder, encode

class Episode:
    def __init__(self):
        self.id=uuid.uuid4().hex;self.config={"data_dir":"/data/yfmo","task_category":"洗衣机"}
        self.task_id="开门";self.policy={"host":"113"}
    def snapshot(self):return {"phase":"running","events":[]}
class Transport:
    def __init__(self):self.fail=False;self.bad_hash=False;self.saved=[]
    def anchor(self):return {"uncertainty_ns":"123"}
    def request(self,method,path,body=None):
        if path=="/api/sessions":return {"success":True,"session":{"raw_directory":"/robot/session/raw"}}
        if path.endswith("/stop"):return {"session":{"collection_state":"stopped"}}
        if self.fail:raise OSError("offline")
        self.saved.append(body["record"])
        return {"success":True,"record_sha256":"wrong" if self.bad_hash else hashlib.sha256(encode(body["record"])).hexdigest()}
class TransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.transport=Transport();self.u=Uploader(self.tmp.name,self.transport)
        self.r=Recorder(Episode(),self.u)
    def test_success_keeps_no_duplicate_payload_on_host(self):
        self.r.write("chunk",{"actions":[[0]*16]})
        self.assertEqual(list(Path(self.tmp.name).glob("*/*.json")),[])
        self.assertEqual(len(self.transport.saved),1)
    def test_failure_retains_retry_cache_and_raises(self):
        self.transport.fail=True
        with self.assertRaises(OSError):self.r.write("chunk",{"actions":[[0]*16]})
        files=list(Path(self.tmp.name).glob("*/*.json"))
        self.assertEqual(len(files),1);self.assertEqual(len(self.u.pending),1)
        self.transport.fail=False;self.u.put(files[0])
        self.assertFalse(files[0].exists())
    def test_hash_mismatch_cannot_acknowledge_record(self):
        self.transport.bad_hash=True
        with self.assertRaises(OSError):self.r.write("chunk",{"actions":[[0]*16]})
        self.assertEqual(len(list(Path(self.tmp.name).glob("*/*.json"))),1)
    def test_nanoseconds_preserved_as_strings(self):
        ns="1794277240123456789";self.r.write("chunk",{"source_ns":ns})
        self.assertEqual(self.transport.saved[0]["data"]["source_ns"],ns)
if __name__=="__main__":unittest.main()
