"""A3 state machine. Pure Python: no robot imports or transport side effects.

`paused` means no NEW dispatch is permitted; it is not a physical-stop acknowledgement.
Per user contract, pause only suppresses subsequent sends; no cancellation/hold RPC.
"""
from collections import deque
import copy
import json
import math
import threading
import time
import uuid

FAILURE_REASONS = {
    'initial_out_of_distribution': '初始状态不适合模型',
    'repetitive_actions': '相似动作循环', 'no_progress': '推进停滞',
    'collision': '碰撞', 'preventive_stop': '预防性终止',
    'grasp_failed': '抓取失败', 'object_slipped': '物体滑落',
    'wrong_pose': '目标位置或姿态错误', 'wrong_stage': '动作阶段错误',
    'unreachable': '限位或不可达', 'unstable_actions': '抖动或发散',
    'runtime_error': '推理、通信或执行异常',
    'intervention_limit': '接管次数用尽', 'manual_reset': '人工重置',
    'other': '其他',
}
TERMINAL = {'succeeded', 'failed'}
VR_FRESH_MS = 250
VR_UNKNOWN_POLICY = 'warn_only'


class Conflict(ValueError):
    pass


class Episode:
    def __init__(self, task_id, config, policy, clock=time.monotonic):
        limit = config['max_interventions']
        if limit is not None and (type(limit) is not int or limit < 0):
            raise ValueError('接管上限必须是非负整数或null')
        self.lock = threading.RLock()
        self.clock = clock
        self.id = uuid.uuid4().hex
        self.task_id = task_id
        self.config = copy.deepcopy(config)
        self.policy = copy.deepcopy(policy)
        self.recording = None
        self.collection_deadline = None
        self.collection_expired = False
        self.created_at = time.time()
        self.started = clock()
        self.phase = 'initializing'
        self.pause_reason = None
        self.revision = 0
        self.generation = 0
        self.rounds = 0
        self.round_open = False
        self.resume_mode = config.get('resume_mode', 'release')
        self.release_debounce_s = .3
        self.release_since = None
        self.release_armed = False
        self.manual_pause_latched = False
        self.limit_exceeded = False
        self.vr = None
        self.vr_at = None
        self.vr_instance = None
        self.last_edge = 0
        self.inflight = None
        self.inference_busy = False
        self.initialization_complete = False
        self.review_started_s = None
        self.review_saved = False
        self.review_payload = None
        self.review_result_queued = False
        self.review_detail = None
        self.result = None
        self.execution_error = None
        self.events = deque(maxlen=2048)
        self.event_count = 0
        self.requests = {}  # bounded below; idempotency within this episode
        self._event('created')

    def _event(self, kind, **data):
        self.revision += 1
        self.event_count += 1
        self.events.append({'seq': self.event_count, 'kind': kind,
                            'elapsed_s': self.clock()-self.started, 'wall_ns':str(time.time_ns()), 'monotonic_ns':str(time.monotonic_ns()), **data})

    def _known_idle(self):
        if not self.vr or self.vr_at is None:
            return False
        age = self.vr.get('age_ms')
        if type(age) not in (int, float) or not math.isfinite(age):
            return False
        return (self.vr.get('fresh') is True and self.vr.get('takeover') is False
                and age >= 0 and age + (self.clock()-self.vr_at)*1000 <= VR_FRESH_MS
                and (self.vr.get('connection') or {}).get('connected') is not False)

    def _pause(self, reason):
        if self.phase in TERMINAL or self.phase in {'awaiting_result', 'resetting'}:
            return
        changed = self.phase != 'paused' or self.pause_reason != reason
        self.phase, self.pause_reason = 'paused', reason
        if changed:
            self.generation += 1  # discard every earlier observation/inference ticket
            self._event('paused', reason=reason, physical_stop_confirmed=False)

    def _takeover(self):
        if self.phase in TERMINAL or self.phase in {'awaiting_result', 'resetting'}:
            return
        if not self.round_open:
            self.rounds += 1
            self.round_open = True
            limit = self.config['max_interventions']
            self.limit_exceeded = limit is not None and self.rounds > limit
            self._event('human_round_started', round=self.rounds, limit_exceeded=self.limit_exceeded)
        self.release_since = None
        self.release_armed = not self.manual_pause_latched
        if not self.manual_pause_latched:
            self._pause('vr_takeover')
        if self.limit_exceeded:
            self.phase = 'awaiting_result'
            self._event('result_required', reason='intervention_limit')

    def observe_vr(self, data):
        with self.lock:
            now = self.clock()
            previous_instance = self.vr_instance
            self.vr, self.vr_at = copy.deepcopy(data), now
            instance = data.get('instance_id')
            if instance and instance != self.vr_instance:
                self.vr_instance = instance
                self.last_edge = max((e.get('seq', 0) for e in data.get('recent_edges', [])), default=0)
                if previous_instance is not None:
                    self.release_armed = False
                    self.release_since = None
                    self._event('vr_observer_restarted', pause_triggered=False)
            # Catch short press/release pairs between polls. Never replay pre-episode history.
            for edge in data.get('recent_edges', []):
                if edge.get('seq', 0) > self.last_edge:
                    self.last_edge = edge['seq']
                    if edge.get('kind') == 'edge':
                        self._event('vr_edge', side=edge.get('side'), active=edge.get('active'))
                        if edge.get('active') is True:
                            self._takeover()
            if data.get('fresh') is True and data.get('takeover') is True:
                self._takeover()
            if not self._known_idle():
                self.release_since = None
                if data.get('fresh') is not True or (data.get('connection') or {}).get('connected') is False:
                    self.release_armed = False
            elif self.release_armed and self.release_since is None:
                self.release_since = now
            self._maybe_release_resume()

    def _maybe_release_resume(self):
        # Only a continuously observed, healthy release of this takeover can resume.
        # Reconnection, manual pause, old inference and errors cannot trigger it.
        if not self._known_idle():
            age = (self.vr or {}).get('age_ms')
            if (not self.vr or self.vr.get('fresh') is not True or
                    (self.vr.get('connection') or {}).get('connected') is False or
                    type(age) not in (int,float) or not math.isfinite(age) or
                    self.vr_at is None or age+(self.clock()-self.vr_at)*1000 > VR_FRESH_MS):
                self.release_armed = False
            self.release_since = None
            return
        if (self.resume_mode != 'release' or not self.release_armed or
                self.release_since is None or self.clock()-self.release_since < self.release_debounce_s or
                self.manual_pause_latched or self.pause_reason != 'vr_takeover' or
                self.phase != 'paused' or not self.initialization_complete or
                self.execution_error or self.limit_exceeded or self.inflight or self.inference_busy):
            return
        self.generation += 1
        self.phase, self.pause_reason = 'running', None
        self.round_open = False
        self.release_armed = False
        self.release_since = None
        self._event('model_resumed', reinitialize=False, trigger='vr_release')

    def watchdog(self):
        with self.lock:
            if (self.collection_deadline is not None and self.clock()>=self.collection_deadline
                    and self.phase not in TERMINAL and self.phase not in {'awaiting_result','resetting'}):
                self.collection_expired = True
                self.generation += 1
                self.phase = 'awaiting_result'
                self._event('result_required',reason='collection_window_exhausted')
            self._maybe_release_resume()
        # Per current user rule, missing/stale VR is diagnostic only during running.
        # Actual takeover edges still invalidate inference and suppress future sends.
        # Start/resume/reset retain a positive fresh-release check.
        return

    def vr_health_warning(self):
        if self._known_idle():
            return None
        if self.vr and self.vr.get('fresh') is True and self.vr.get('takeover') is True:
            return None  # a detected takeover is represented by the actual pause state
        return (self.vr or {}).get('reason') or 'vr_state_unknown'

    def begin_initialization(self):
        with self.lock:
            if self.phase != 'initializing' or not self._known_idle():
                raise Conflict('初始化前需要有效且松开的VR状态')
            return self._begin_dispatch('initialization', self.generation)

    def initialized(self, token, success=True):
        with self.lock:
            if self.inflight != token:
                return
            self.inflight = None
            if not success:
                self.worker_failed('initialization_failed')
            elif self.phase == 'initializing' and token[1] == self.generation:
                self.initialization_complete = True
                self.phase = 'running'
                self._event('model_started')

    def begin_inference(self):
        with self.lock:
            self.watchdog()
            if self.phase != 'running' or self.inflight or self.inference_busy:
                return None
            self.inference_busy = True
            return (self.id, self.generation)

    def _begin_dispatch(self, kind, generation):
        if self.inflight:
            raise Conflict('上一执行请求尚未返回')
        self.inflight = (self.id, generation, uuid.uuid4().hex, kind)
        self._event('dispatch_started', dispatch_kind=kind)
        return self.inflight

    def accept_inference(self, ticket):
        with self.lock:
            self.inference_busy = False
            self.watchdog()
            if ticket != (self.id, self.generation) or self.phase != 'running':
                self._event('inference_discarded')
                return None
            return self._begin_dispatch('model_chunk', ticket[1])

    def send_dispatch(self, token, send):
        # Serialize only the transport send with pause, never the acknowledgement wait.
        # A pause accepted first prevents this send. A send accepted first is in-flight
        # and is not cancelled; no stop or hold command is emitted.
        with self.lock:
            self.watchdog()
            if token != self.inflight:
                return False
            if token[1] != self.generation or self.phase != 'running':
                self.inflight = None
                self._event('dispatch_discarded')
                return False
            send()
            self._event('transport_sent')
            return True

    def dispatch_done(self, token, success=True):
        with self.lock:
            if token != self.inflight:
                return
            self.inflight = None
            self._event('dispatch_returned', accepted=bool(success), physical_stop_confirmed=False)
            if not success:
                self.worker_failed('execute_failed')

    def inference_failed(self, ticket, error):
        with self.lock:
            self.inference_busy = False
            if ticket != (self.id, self.generation) or self.phase != 'running':
                self.execution_error=str(error)[:1000]
                self._event('inference_discarded', error=self.execution_error, reconnect_required=True)
                return
            self.worker_failed(error)

    def worker_failed(self, error):
        with self.lock:
            self.inference_busy = False
            if self.phase in TERMINAL:
                return
            self.execution_error = str(error)[:1000]
            self.generation += 1
            self.phase = 'awaiting_result'
            self._event('result_required', reason='runtime_error', error=self.execution_error)

    def natural_end(self):
        with self.lock:
            if self.phase in TERMINAL:
                return
            self.generation += 1
            self.phase = 'awaiting_result'
            self._event('result_required', reason='model_finished')

    def command(self, command, episode_id, revision, request_id, reasons=None, note=''):
        with self.lock:
            if episode_id != self.id:
                raise Conflict('旧任务页面，请刷新')
            if not isinstance(request_id, str) or not 1 <= len(request_id) <= 100:
                raise ValueError('缺少请求标识')
            if not isinstance(note, str) or len(note) > 2000:
                raise ValueError('说明最长2000字')
            if reasons is not None and (not isinstance(reasons, list) or any(not isinstance(x,str) for x in reasons)):
                raise ValueError('失败原因必须是字符串列表')
            if command == 'cancel' and (type(revision) is not int or not 0 <= revision <= self.revision):
                raise Conflict('取消任务的版本无效')
            signature = (command, tuple(reasons or []), note)
            if request_id in self.requests:
                if self.requests[request_id] != signature:
                    raise Conflict('请求标识重复但内容不同')
                return self.snapshot()
            if command != 'cancel' and revision != self.revision:
                raise Conflict('任务状态已变化，请重新确认当前状态')
            reopening_review=command=='review' and self.review_started_s is not None and not self.review_saved
            if self.phase in TERMINAL and not reopening_review:
                raise Conflict('任务已结束')
            if command == 'pause':
                self.manual_pause_latched = True
                self.release_armed = False
                self.release_since = None
                self._pause('manual_interrupt')
            elif command == 'review':
                if (self.phase not in {'paused','awaiting_result'} and not reopening_review) or self.inflight:
                    raise Conflict('请先中止任务，并等待正在发送的动作请求结束')
                if self.review_started_s is None:
                    self.review_started_s = self.clock()-self.started
                    self.generation += 1
                    self.phase = 'awaiting_result'
                    self.manual_pause_latched = True
                    self.release_armed = False
                    self.round_open = False
                    self._event('review_started', outcome='failure')
            elif command == 'resume':
                if (self.execution_error or not self.initialization_complete or self.phase != 'paused' or self.limit_exceeded or self.inflight
                        or self.inference_busy or not self._known_idle()):
                    raise Conflict('需松开接管键、恢复新数据并等待旧请求结束后才能继续')
                self.generation += 1
                self.phase, self.pause_reason = 'running', None
                self.round_open = False
                self.manual_pause_latched = False
                self.release_armed = False
                self.release_since = None
                self._event('model_resumed', reinitialize=False, trigger='web')
            elif command in {'success', 'failure', 'reset', 'cancel'}:
                if command == 'cancel':
                    if self.review_started_s is not None:
                        raise Conflict('本条已进入失败标注，请保存本条数据')
                    selected = ['preventive_stop']
                elif self.phase not in {'paused', 'awaiting_result'} or self.inflight:
                    raise Conflict('请先中断，并等待正在返回的执行请求结束')
                else:
                    selected = list(dict.fromkeys(reasons or []))
                if command == 'reset':
                    if self.execution_error:
                        raise Conflict('通信或执行异常：请先标注结果，结束任务后再独立复位')
                    if self.inference_busy or not self._known_idle():
                        raise Conflict('复位前需松开接管键，并等待旧推理结束')
                    selected = ['manual_reset']
                if command == 'failure':
                    if not selected or any(x not in FAILURE_REASONS for x in selected):
                        raise ValueError('失败需至少选择一个有效原因')
                    if 'other' in selected and not note.strip():
                        raise ValueError('选择其他时需填写说明')
                if command == 'success' and self.limit_exceeded:
                    raise Conflict('接管次数已超限，本次需标注失败')
                if not isinstance(note, str) or len(note) > 2000:
                    raise ValueError('说明最长2000字')
                self.generation += 1
                self.result = {'success': command == 'success', 'reasons': selected,
                               'note': note, 'duration_s': self.review_started_s if self.review_started_s is not None else self.clock()-self.started,
                               'interventions': self.rounds,
                               'model_independent_success': command == 'success' and self.rounds == 0}
                if command == 'cancel':
                    self.result['cancelled'] = True
                self.phase = 'resetting' if command == 'reset' else ('succeeded' if command == 'success' else 'failed')
                self._event('result_decided', result=copy.deepcopy(self.result))
            else:
                raise ValueError('未知操作')
            self.requests[request_id] = signature
            if len(self.requests) > 256:
                self.requests.pop(next(iter(self.requests)))
            return self.snapshot()

    def reset_done(self, success):
        with self.lock:
            if self.phase != 'resetting':
                return
            self.phase = 'failed'  # original episode remains a failure even after successful reset
            self._event('reset_returned', accepted=bool(success))

    def snapshot(self):
        with self.lock:
            return {'id': self.id, 'taskId': self.task_id, 'phase': self.phase,
                    'review_started_s':self.review_started_s,'review_pending':self.review_started_s is not None and not self.review_saved,'review_saved':self.review_saved,
                    'review_submission':json.loads(self.review_payload) if self.review_payload and not self.review_saved else None,
                    'revision': self.revision, 'generation': self.generation,
                    'vr_unknown_policy':VR_UNKNOWN_POLICY, 'vr_health_warning':self.vr_health_warning(),
                    'resume_mode':self.resume_mode, 'release_debounce_ms':300, 'pause_reason': self.pause_reason, 'execution_error':self.execution_error, 'interventions': self.rounds,
                    'max_interventions': self.config['max_interventions'],
                    'limit_exceeded': self.limit_exceeded,
                    'inflight': self.inflight is not None, 'inference_busy': self.inference_busy,
                    'can_resume': not self.execution_error and self.initialization_complete and self.phase == 'paused' and self._known_idle()
                                  and not self.inflight and not self.inference_busy and not self.limit_exceeded,
                    'physical_stop_confirmed': False, 'result': copy.deepcopy(self.result),
                    'policy': copy.deepcopy(self.policy), 'elapsed_s': self.review_started_s if self.review_started_s is not None else self.clock()-self.started,
                    'events': list(copy.deepcopy(self.events)), 'event_count': self.event_count,
                    'persistent_recording': self.recording is not None,
                    'recording':self.recording.status() if self.recording else None}


class Worker:
    """Dependency-injected execution loop; caller owns monitoring and persistence.

    Adapter initialization is called exactly once. Resume takes a NEW observation.
    `dispatch` is an executor-acceptance boundary, not a motion-completion signal.
    """
    def __init__(self, episode, adapter):
        self.episode, self.adapter = episode, adapter

    def initialize(self):
        token = self.episode.begin_initialization()
        try:
            ok = self.adapter.initialize()
        except Exception as exc:
            self.episode.initialized(token, False)
            return
        self.episode.initialized(token, ok is not False)

    def step(self):
        ticket = self.episode.begin_inference()
        if ticket is None:
            return False
        token = None
        try:
            observation = self.adapter.observe()
            chunk = self.adapter.infer(observation)
            token = self.episode.accept_inference(ticket)
            if token is None:
                if hasattr(self.adapter,'record_outcome'):self.adapter.record_outcome('discarded_before_dispatch')
                return False
            try:
                prepared = self.adapter.prepare_dispatch(chunk)
                if not self.episode.send_dispatch(token, lambda: self.adapter.send_prepared(prepared)):
                    if hasattr(self.adapter,'record_outcome'):self.adapter.record_outcome('discarded_before_send')
                    return False
                ok = self.adapter.receive_ack()
            except Exception:
                self.episode.dispatch_done(token, False)
                if hasattr(self.adapter,'record_outcome'):self.adapter.record_outcome('send_or_ack_error')
                raise
            self.episode.dispatch_done(token, ok is not False)
            return True
        except Exception as exc:
            if token is None:
                self.episode.inference_failed(ticket, exc)
            else:
                self.episode.worker_failed(exc)
            return False
