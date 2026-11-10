import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
import uuid
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"robot"))
from collection_server import handler_for
from storage import SessionStore
from record_view import build_episode
from test_sessions import Collector

class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.store=SessionStore(self.tmp.name,Collector(),build_episode)
        self.token="t"*64
        ready=lambda c:{"ready":True,"robot_ready":True,"vr":{"fresh":True,"takeover":False}}
        self.server=ThreadingHTTPServer(("127.0.0.1",0),handler_for(self.store,self.token,ready))
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.addCleanup(self.close)
    def close(self):self.server.shutdown();self.server.server_close();self.thread.join()
    def call(self,path,body=None,auth=True):
        headers={"Authorization":"Bearer "+self.token} if auth else {}
        r=Request("http://127.0.0.1:"+str(self.server.server_port)+path,
            data=json.dumps(body).encode() if body is not None else None,headers=headers)
        with urlopen(r,timeout=3) as response:return json.load(response)
    def test_unauthorized_cannot_read_or_start(self):
        with self.assertRaises(HTTPError) as exc:self.call("/api/health",auth=False)
        self.assertEqual(exc.exception.code,401)
        self.assertEqual(list(self.store.index.glob("*.json")),[])
    def test_start_returns_confirmed_robot_local_session(self):
        sid=uuid.uuid4().hex
        r=self.call("/api/sessions",{"episode_id":sid,"task_id":"开门","category":"洗衣机"})
        self.assertEqual(r["session"]["collection_state"],"recording")
        self.assertEqual(len(self.store.collector.starts),1)
        self.call("/api/sessions",{"episode_id":sid,"task_id":"开门","category":"洗衣机"})
        self.assertEqual(len(self.store.collector.starts),1)
        self.call("/api/sessions/"+sid+"/stop",{})
        self.assertEqual(self.store.manifest(sid)["collection_state"],"stopped")
    def test_health_does_not_start_recording(self):
        self.call("/api/health")
        self.assertEqual(self.store.collector.starts,[])
    def test_not_ready_start_has_no_files(self):
        self.server.RequestHandlerClass=handler_for(self.store,self.token,lambda c:{"ready":True,"robot_ready":True,"vr":{"fresh":False}})
        with self.assertRaises(HTTPError):self.call("/api/sessions",{"episode_id":uuid.uuid4().hex,"task_id":"开门","category":"洗衣机"})
        self.assertEqual(list(self.store.index.glob("*.json")),[])
if __name__=="__main__":unittest.main()
