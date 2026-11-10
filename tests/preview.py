"""Local UI verification with a fake collector; never reaches a robot."""
import sys
from pathlib import Path
from http.server import ThreadingHTTPServer
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"host"),str(ROOT/"robot"),str(ROOT/"tests")]
from test_sessions import SessionTests
import intergrated_test_server as app
fixture=SessionTests();fixture.setUp();fixture.three_rounds();fixture.finish()
app.vr_proxy.fetch_status=lambda:(200,{"success":True,"fresh":True,"takeover":False,"age_ms":0})
class Records:
    def listing(self,params):return fixture.store.list_sessions({k:v[0] for k,v in params.items()})
    def detail(self,sid):return {"success":True,"episode":fixture.store.detail(sid)}
    def chunk(self,*args):raise FileNotFoundError("fixture has no model chunk")
    def write(self,sid,op,body):
        if op=="annotations":result=fixture.store.annotate(sid,body["round"],body.get("reasons",[]),body.get("note",""),body["revision"])
        elif op=="finalize":result=fixture.store.finalize(sid,body["revision"])
        else:result=fixture.store.stop(sid)
        return {"success":True,"episode":result}
app.records=Records()
server=ThreadingHTTPServer(("127.0.0.1",18089),app.IntegratedHandler)
print("PREVIEW_READY",fixture.sid,flush=True)
try:server.serve_forever()
finally:server.server_close();fixture.tmp.cleanup()
