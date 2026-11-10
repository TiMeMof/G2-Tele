import sys
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"host"))
from episode_control import Episode, Conflict

class Clock:
    def __init__(self):self.t=0.
    def __call__(self):return self.t

class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.clock=Clock()
        self.e=Episode("开门",{"max_interventions":3},{"host":"113"},self.clock)
        self.feed(False)
        token=self.e.begin_initialization();self.e.initialized(token,True)
    def feed(self,active,fresh=True,connected=True,instance="one"):
        self.e.observe_vr({"fresh":fresh,"takeover":active,"age_ms":0,
            "instance_id":instance,"recent_edges":[],"connection":{"connected":connected}})
    def release(self):
        self.clock.t+=.01;self.feed(False)
        for _ in range(4):self.clock.t+=.1;self.feed(False)
    def command(self,name):
        return self.e.command(name,self.e.id,self.e.revision,name+"-"+str(self.e.revision))
    def test_stable_release_automatically_closes_one_round(self):
        self.feed(True);self.release()
        self.assertEqual(self.e.phase,"running");self.assertEqual(self.e.rounds,1)
        self.assertFalse(self.e.round_open)
        self.assertEqual([e for e in self.e.events if e["kind"]=="model_resumed"][-1]["trigger"],"vr_release")
    def test_short_release_does_not_resume(self):
        self.feed(True);self.clock.t+=.01;self.feed(False)
        self.clock.t+=.1;self.feed(True)
        self.assertEqual(self.e.phase,"paused");self.assertEqual(self.e.rounds,1)
    def test_each_complete_takeover_creates_its_own_interval(self):
        for _ in range(3):self.feed(True);self.release()
        self.assertEqual(self.e.rounds,3)
        self.assertEqual(len([e for e in self.e.events if e["kind"]=="model_resumed"]),3)
    def test_one_hand_still_active_blocks_resume(self):
        self.feed(True)
        for _ in range(8):self.clock.t+=.1;self.feed(True)
        self.assertEqual(self.e.phase,"paused")
    def test_disconnect_never_counts_as_release(self):
        self.feed(True);self.feed(None,fresh=False,connected=False);self.release()
        self.assertEqual(self.e.phase,"paused")
    def test_gap_without_poll_also_blocks_automatic_resume(self):
        self.feed(True);self.feed(False)
        self.clock.t+=.5;self.e.watchdog();self.release()
        self.assertEqual(self.e.phase,"paused")
    def test_observer_restart_requires_explicit_resume(self):
        self.feed(True);self.feed(False,instance="two");self.release()
        self.assertEqual(self.e.phase,"paused")
    def test_manual_pause_stays_manual_after_vr_activity(self):
        self.command("pause");self.feed(True);self.release()
        self.assertEqual(self.e.phase,"paused");self.assertTrue(self.e.manual_pause_latched)
        self.command("resume");self.assertEqual(self.e.phase,"running")
    def test_old_inference_is_discarded_before_new_observation(self):
        ticket=self.e.begin_inference();self.feed(True);self.release()
        self.assertEqual(self.e.phase,"paused")
        self.assertIsNone(self.e.accept_inference(ticket))
        self.assertEqual(self.e.phase,"running")
        self.assertNotEqual(self.e.begin_inference(),ticket)
    def test_inflight_send_blocks_resume(self):
        ticket=self.e.begin_inference();token=self.e.accept_inference(ticket)
        self.feed(True);self.release();self.assertEqual(self.e.phase,"paused")
        self.e.dispatch_done(token,True);self.e.watchdog()
        self.assertEqual(self.e.phase,"running")
    def test_errors_and_intervention_limit_do_not_resume(self):
        self.feed(True);self.e.worker_failed("network_error");self.release()
        self.assertEqual(self.e.phase,"awaiting_result")
    def test_limit_exceeded_stays_awaiting_result(self):
        for _ in range(4):self.feed(True);self.release()
        self.assertTrue(self.e.limit_exceeded);self.assertEqual(self.e.phase,"awaiting_result")
    def test_stale_release_cannot_send_a_chunk(self):
        self.feed(True);self.feed(False);self.clock.t+=1;self.e.watchdog()
        self.assertIsNone(self.e.begin_inference())
    def test_collection_window_expiry_discards_late_prediction(self):
        self.e.collection_deadline=.2
        ticket=self.e.begin_inference()
        self.clock.t=.3;self.e.watchdog()
        self.assertTrue(self.e.collection_expired)
        self.assertEqual(self.e.phase,"awaiting_result")
        self.assertIsNone(self.e.accept_inference(ticket))
if __name__=="__main__":unittest.main()
