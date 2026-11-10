"""Runtime: cease future sends on takeover; robot feedback is collected externally."""
import threading
import time
from pathlib import Path
from chunk_log import Recorder, shared_uploader
from washer_adapter import WasherAdapter
from washer_tasks import TASK_IDS, policy_for, resolve_running_policy, get_policy_host
from episode_control import Episode, Worker, Conflict, FAILURE_REASONS, TERMINAL, VR_FRESH_MS, VR_UNKNOWN_POLICY

POLICY = {
    'host': get_policy_host(), 'port': 8119, 'config': 'linden_lingao',
    'checkpoint': '/home/lindenbot/qianzhang43/checkpoints/yfmo_washer_close_the_gate_17.5k_0922',
    'asset_id': 'washer_close_the_gate_0920', 'action_shape': [30, 16],
    'task_id': '双臂关洗衣机门', 'action_space': 'joint_absolute',
}


_health_cache = {"at":0.0,"value":{}}
_health_lock = threading.Lock()

def collection_health():
    with _health_lock:
        if time.monotonic()-_health_cache["at"]>1:
            try:
                _health_cache["value"]=shared_uploader().transport.request('GET','/api/health')
            except OSError as exc:
                _health_cache["value"]={"ready":False,"message":str(exc)}
            _health_cache["at"]=time.monotonic()
        return dict(_health_cache["value"])

def readiness():
    configured = Path(__file__).with_name('session_receiver.token').is_file()
    health=collection_health() if configured else {}
    vr=health.get('vr',{})
    ready=(configured and health.get('ready') is True and health.get('robot_ready') is True
           and vr.get('fresh') is True and vr.get('takeover') is False)
    return {'ready': ready, 'robot_collection':health, 'state_machine_tested': True,
            'pause_policy': 'suppress_future_sends_only', 'vr_unknown_policy':VR_UNKNOWN_POLICY, 'physical_handover_verified': False,
            'persistent_recording_ready': configured, 'feedback_collection': 'robot_unified_session',
            'blockers': [] if ready else [health.get('message') or ('请先连接VR并松开接管键' if health.get('ready') else '机器人统一采集服务尚未就绪')],
            'policy': POLICY, 'supported_task_ids':list(TASK_IDS), 'failure_reasons': FAILURE_REASONS}


def make_recorder(episode):
    uploader = shared_uploader()
    uploader.transport.request('POST','/api/probe',{'data_dir':episode.config['data_dir']})
    uploader.anchor_value=uploader.transport.anchor();uploader.anchor_at=time.monotonic()
    recorder = Recorder(episode,uploader)
    try:
        recorder.write('metadata',{'task_id':episode.task_id,'config':episode.config,'policy':episode.policy,
            'pause_policy':'suppress_future_sends_only','vr_threshold':0.2,'vr_stale_ms':VR_FRESH_MS,
            'vr_unknown_policy':VR_UNKNOWN_POLICY,'resume_mode':episode.resume_mode,
            'feedback_source':'robot_unified_session',
            'clock_domains':'host and robot wall/monotonic plus native source timestamps',
            'continuous_joint_or_vr_recording':True,'frame_execution_timestamps_available':False})
    except Exception:
        try: recorder.stop_collection()
        except Exception: pass
        raise
    return recorder


class Runtime:
    def __init__(self, vr_reader, adapter_factory=None, readiness_check=readiness, recorder_factory=None):
        self.recorder_factory = recorder_factory
        self.recording_error = None
        self.lock = threading.RLock()
        self.vr_reader = vr_reader
        self.adapter_factory = adapter_factory
        self.readiness_check = readiness_check
        self.episode = None
        self.stop = threading.Event()
        self.worker_thread = None
        self.monitor_thread = None
        self.writer_thread = None

    def active(self):
        with self.lock:
            return self.episode is not None and self.episode.phase not in TERMINAL

    def status(self):
        with self.lock:
            return {'success': True, 'readiness': self.readiness_check(),
                    'episode': self.episode.snapshot() if self.episode else None, 'recording_error':self.recording_error}

    def _read_vr(self, episode):
        try:
            code, data = self.vr_reader()
            if code != 200:
                data = {'fresh': False, 'takeover': None, 'reason': data.get('reason','unavailable')}
        except Exception:
            data = {'fresh': False, 'takeover': None, 'reason': 'unavailable'}
        episode.observe_vr(data)

    def start(self, task_id, config):
        with self.lock:
            ready = self.readiness_check()
            if not ready['ready'] or self.adapter_factory is None:
                raise Conflict('人工接管模式暂未开放：'+'；'.join(ready['blockers']))
            try:
                policy = resolve_running_policy(policy_for(task_id))
            except (ValueError, OSError) as exc:
                raise Conflict(str(exc)) from exc
            if self.active() or (self.worker_thread and self.worker_thread.is_alive()):
                raise Conflict('上一任务或请求尚未结束')
            config = {**config, 'task_category':'洗衣机', 'resume_mode':'release'}
            episode = Episode(task_id, config, policy)
            self._read_vr(episode)
            if not episode._known_idle() or episode.phase != 'initializing':
                raise Conflict('开始前需要新鲜且未接管的VR数据')
            if self.recorder_factory:
                episode.recording = self.recorder_factory(episode)
                episode.recording.flush_events()
            self.episode = episode
            self.stop = threading.Event()
            self.monitor_thread = threading.Thread(target=self._monitor, args=(episode,self.stop), daemon=True)
            self.writer_thread = threading.Thread(target=self._write_events, args=(episode,self.stop), daemon=True)
            self.worker_thread = threading.Thread(target=self._worker, args=(episode,self.stop), daemon=True)
            self.monitor_thread.start()
            self.writer_thread.start()
            self.worker_thread.start()
            return self.status()

    def _monitor(self, episode, stop):
        while not stop.is_set():
            self._read_vr(episode)
            episode.watchdog()
            stop.wait(.05)

    def _write_events(self, episode, stop):
        while not stop.is_set():
            try:
                if episode.recording:episode.recording.flush_events()
            except Exception as exc:
                self.recording_error=str(exc)
                episode.worker_failed('事件记录失败：'+str(exc))
            stop.wait(.1)

    def _worker(self, episode, stop):
        adapter = None
        try:
            adapter = self.adapter_factory(episode)
            worker = Worker(episode, adapter)
            # The start request verified a released VR frame, but the asynchronous
            # worker may arrive after its 250 ms diagnostic freshness window.
            # Wait for a new released frame before any initialization command.
            # A real takeover or manual interrupt exits without initializing.
            while not stop.is_set():
                with episode.lock:
                    if episode.phase != 'initializing':
                        return
                    if episode._known_idle():
                        break
                stop.wait(.02)
            if stop.is_set():
                return
            worker.initialize()
            while not stop.is_set():
                phase = episode.snapshot()['phase']
                if phase in TERMINAL or episode.collection_expired:
                    break
                if phase == 'resetting':
                    try:episode.reset_done(adapter.reset() is not False)
                    except Exception:episode.reset_done(False)
                    continue
                episode.watchdog()
                if not worker.step():
                    stop.wait(.02)
        except Exception as exc:
            episode.worker_failed(exc)
        finally:
            # A runtime exception remains available for manual result classification.
            # Monitoring must remain alive until that classification is complete.
            if adapter is not None:
                try:adapter.close()
                except Exception:pass
            try:
                if episode.recording:
                    episode.recording.flush_events()
                    episode.recording.write('session_end',{'phase':episode.phase,'result':episode.result,'worker_exited':True})
                    episode.recording.stop_collection()
            except Exception as exc:
                self.recording_error=str(exc)
                try:
                    if episode.recording: episode.recording.stop_collection()
                except Exception as stop_error: self.recording_error += '; ' + str(stop_error)
            if episode.phase in TERMINAL:
                stop.set()

    def command(self, body):
        with self.lock:
            if self.episode is None:
                raise Conflict('当前没有人工接管任务')
            if body.get('command') in {'resume', 'reset'}:
                self._read_vr(self.episode)
                if not self.worker_thread or not self.worker_thread.is_alive():
                    raise Conflict('执行器已退出；请标注结果，不能直接继续或复位')
            with self.episode.lock:
                result = self.episode.command(body.get('command'),body.get('episodeId'),body.get('revision'),
                                              body.get('requestId'),body.get('reasons'),body.get('note',''))
                try:
                    if self.episode.recording:
                        if body.get('command') in {'success','failure','reset'}:
                            self.episode.recording.write('result',{'result':self.episode.result,'command':body.get('command'),'request_id':body.get('requestId')})
                        self.episode.recording.flush_events()
                except Exception as exc:
                    self.recording_error=str(exc)
                    self.episode.result=None
                    self.episode.requests.pop(body.get('requestId'),None)
                    self.episode.phase='awaiting_result'
                    self.episode.worker_failed('结果保存失败：'+str(exc))
                    raise OSError('结果未保存，已阻止继续/复位：'+str(exc))
            if self.episode.phase in TERMINAL:
                self.stop.set()
            return {'success': True, 'episode': result}
