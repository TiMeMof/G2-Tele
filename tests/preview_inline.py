"""Local-only two-round failure dialog fixture. No robot APIs or model clients."""
import json
from pathlib import Path
import sys
import threading
from http.server import ThreadingHTTPServer
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'host'),str(ROOT/'robot'),str(ROOT/'tests')]
from test_inline_review import InlineReviewTests
from episode_control import FAILURE_REASONS
import intergrated_test_server as app

fixture=InlineReviewTests();fixture.setUp()
fixture.ep.task_id='右臂开洗衣机门'
app.takeover=fixture.runtime
app.takeover.readiness_check=lambda:{'ready':True,'persistent_recording_ready':True,
    'failure_reasons':FAILURE_REASONS,'supported_task_ids':['右臂开洗衣机门']}
app.vr_proxy.fetch_status=lambda:(200,{'success':True,'fresh':True,'takeover':False,'age_ms':0})

def finish_worker():
    fixture.runtime.stop.wait()
    fixture.ep.recording.flush_events()
    fixture.ep.recording.write('session_end',{'phase':fixture.ep.phase,'result':None})
    fixture.ep.recording.stop_collection(refresh_barrier=True)
fixture.runtime.worker_thread=threading.Thread(target=finish_worker,daemon=True)
fixture.runtime.worker_thread.start()

class Records:
    def listing(self,params):return fixture.store.list_sessions({k:v[0] for k,v in params.items()})
    def detail(self,sid):return {'success':True,'episode':fixture.store.detail(sid)}
    def chunk(self,*args):raise FileNotFoundError('no model chunks in fixture')
app.records=Records()

class Handler(app.IntegratedHandler):
    def do_GET(self):
        if self.path=='/api/test-state':
            value=fixture.store.manifest(fixture.ep.id)
            self.send_json(200,{'result':value['result'],'annotations':value['round_annotations'],
                'archive_state':value['archive_state'],'collection_state':value['collection_state'],
                'stop_count':len(fixture.collector.stops)})
        else:super().do_GET()
server=ThreadingHTTPServer(('127.0.0.1',18090),Handler)
print('INLINE_PREVIEW_READY',flush=True)
try:server.serve_forever()
finally:server.server_close();fixture.tmp.cleanup()
