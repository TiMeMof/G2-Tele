"""Washer adapter with timestamped action chunks.

Only explicit runtime start may instantiate this class. No automatic mode changes,
remote_env start/stop, or implicit reset in the constructor. Pause sends no stop,
hold or cancellation requests; it only prevents subsequent model sends.
"""
import json
import logging
import threading
from joint_limits import JointLimits
import time
from chunk_log import stamp
from pathlib import Path
from network_config import get_robot_host


class WasherAdapter:
    def __init__(self, episode, root=None):
        import cv2
        import msgpack
        import numpy as np
        from websockets.sync.client import connect
        from openpi_client import msgpack_numpy
        import urllib.request
        self.np,self.cv2,self.msgpack,self.codec=np,cv2,msgpack,msgpack_numpy
        self.episode=episode
        self.recorder=episode.recording
        self.chunk_id=None
        self.observation_reference={}
        self.sent_times=None
        root=Path(root) if root else Path(__file__).resolve().parent.parent
        self.task=json.loads((root/'config/config_V2_0.json').read_text(encoding='utf-8'))[episode.task_id]
        self.joint_limits=JointLimits(Path(__file__).resolve().with_name('joint_limits.json'))
        self.last_clip_warning=None
        self.robot=None;self.robot_actions=None;self.model=None
        self._action_reader_stop=threading.Event()
        self._action_reader=None
        self._action_reader_error=None
        self._ignored_action_replies=0
        state=json.load(urllib.request.urlopen(f'http://{get_robot_host()}:8890/system/status',timeout=2))
        if not state.get('success') or not state['data']['instance_running'] or state['data']['current_mode']!='remote_env':
            raise RuntimeError('226已有remote_env实例未就绪；不会自动启动或切换模式')
        try:
            self.robot=connect(f'ws://{get_robot_host()}:8891',compression=None,open_timeout=3,close_timeout=2,max_size=20000000)
            self.model=connect(f"ws://{episode.policy['host']}:{episode.policy['port']}",compression=None,open_timeout=3,close_timeout=2,max_size=20000000)
            self.metadata=msgpack_numpy.unpackb(self.model.recv(timeout=3))
            self.robot_actions=connect(f'ws://{get_robot_host()}:8891',compression=None,open_timeout=3,close_timeout=2,max_size=20000000)
            self._action_reader=threading.Thread(target=self._drain_action_replies,daemon=True)
            self._action_reader.start()
        except Exception:
            self.close();raise

    def _drain_action_replies(self):
        # Drain replies on a dedicated socket without gating the next chunk.
        while not self._action_reader_stop.is_set():
            try:self.robot_actions.recv(timeout=.5)
            except TimeoutError:continue
            except Exception as exc:
                if not self._action_reader_stop.is_set():self._action_reader_error=exc
                return
            self._ignored_action_replies+=1

    def _rpc(self, method, params):
        self.robot.send(self.msgpack.packb({'method':method,'params':params}))
        reply=self.msgpack.unpackb(self.robot.recv(timeout=5),raw=False)
        if reply.get('error'):raise RuntimeError(str(reply['error']))
        return reply.get('result')

    def observe(self):
        before=stamp()
        obs=self._rpc('get_observation',{})
        after=stamp()
        self.observation_reference={'request_114':before,'received_114':after,
            'timestamps_raw':{str(k):str(v) for k,v in obs.get('timestamps',{}).items()},
            'image_timestamp_fields':{k:{x:str(v[x]) for x in v if 'time' in x} for k,v in obs.get('images',{}).items() if isinstance(v,dict)},
            'source_clock':'robot_observation_native_unconverted'}
        a=obs['states']['arm_joint_states'];g=obs['states']['gripper_states']
        state=list(a[:7])+[g[0]]+list(a[7:14])+[g[1]]
        if len(state)!=16 or not self.np.isfinite(state).all():raise ValueError('无效关节状态')
        images={}
        for src,key in [('head','cam_head'),('hand_left','cam_left'),('hand_right','cam_right')]:
            image=obs['images'][src]
            if image['encoding']!='JPEG':raise ValueError('不支持的相机编码')
            images[key]=self.cv2.imdecode(self.np.frombuffer(image['image_data'],dtype=self.np.uint8),self.cv2.IMREAD_COLOR)
            if images[key] is None:raise ValueError('图像解码失败')
        return {'state':state,'images':images,'prompt':self.task['prompt']}

    def infer(self, observation):
        self.chunk_id=__import__('uuid').uuid4().hex
        self.sent_times=None
        started=stamp()
        self.model.send(self.codec.Packer().pack(observation))
        raw=self.model.recv(timeout=40)
        finished=stamp()
        if isinstance(raw,str):raise RuntimeError(raw[:1000])
        response=self.codec.unpackb(raw)
        actions=self.np.asarray(response['actions'])
        if actions.shape!=(30,16) or not self.np.isfinite(actions).all():raise ValueError('模型动作块必须为有限的30x16')
        if self.recorder:
            self.recorder.write('chunk',{'chunk_id':self.chunk_id,'actions':actions.tolist(),'dtype':str(actions.dtype),'shape':list(actions.shape),
                'inference_start_114':started,'inference_end_114':finished,'observation':self.observation_reference,
                'input_state':list(observation['state']),'prompt':observation['prompt'],
                'policy_timing':response.get('policy_timing'),
                'status':'predicted_not_yet_sent','trajectory_reference_time_s':1.0,
                'frame_times_are_predictions_not_actual_execution':True})
        return actions

    def prepare_dispatch(self, chunk):
        clipped, clipping=self.joint_limits.clip(chunk)
        left=[row[:7] for row in clipped]
        right=[row[8:15] for row in clipped]
        action={'trajectory_reference_time':1.0,'base_link':'base_link',
                'left_arm':{'kind':'JOINT_ABS','values':left},
                'right_arm':{'kind':'JOINT_ABS','values':right},
                'left_effector':chunk[:,7:8].tolist(),'right_effector':chunk[:,15:16].tolist()}
        if self.recorder:self.recorder.write('dispatch',{'chunk_id':self.chunk_id,'status':'intent_not_sent','prepared_114':stamp(),'command':action,'joint_limit_clipping':clipping})
        if clipping['applied']:
            now=time.monotonic()
            if self.last_clip_warning is None or now-self.last_clip_warning>=5.0:
                logging.warning('Model joint targets clipped: chunk=%s values=%d max_delta_rad=%.6f limits=%s',
                                self.chunk_id,clipping['changed_values'],clipping['max_abs_delta_rad'],clipping['version'])
                self.last_clip_warning=now
        return self.msgpack.packb({'method':'execute_action','params':action})

    def send_prepared(self, payload):
        self.sent_times={'send_start_114':stamp()}
        try:
            if self._action_reader_error is not None:
                raise ConnectionError('动作发送连接已断开') from self._action_reader_error
            self.robot_actions.send(payload)
            self.sent_times['send_return_114']=stamp()
        except Exception:
            self.sent_times['send_error_114']=stamp()
            raise

    def record_outcome(self,status,**extra):
        if self.recorder:self.recorder.write('dispatch',{'chunk_id':self.chunk_id,'status':status,'at_114':stamp(),**(self.sent_times or {}),**extra})

    def receive_ack(self):
        # Worker asks whether transport send completed; do not inspect RPC result.
        self.record_outcome('transport_sent',ack_policy='not_waited')
        return True

    def initialize(self):
        # An explicit start would initialize once; resume never reaches this method.
        return self.reset()

    def reset(self):
        positions=self.task['init_position']
        if isinstance(positions,dict):positions=[positions]
        for position in positions:
            # Multi-stage reset must also yield to a takeover. No blind follow-up stage.
            self.episode.watchdog()
            with self.episode.lock:
                if not self.episode._known_idle() or self.episode.phase not in {'initializing','resetting'}:
                    return False
            arm=position['arm'];grip=position['gripper']
            action={'trajectory_reference_time':2.0,'base_link':'base_link',
                    'left_arm':{'kind':'JOINT_ABS','values':[arm[:7]]},
                    'right_arm':{'kind':'JOINT_ABS','values':[arm[7:]]},
                    'left_effector':[[grip[0]]],'right_effector':[[grip[1]]],
                    'head':[position['head']], 'waist':{'kind':'ABS_JOINT','values':[position['waist']]}}
            if self.recorder:self.recorder.write('initialization',{'status':'intent_not_sent','command':action,'at_114':stamp()})
            with self.episode.lock:
                if not self.episode._known_idle() or self.episode.phase not in {'initializing','resetting'}:return False
                started=stamp()
                if self._action_reader_error is not None:
                    raise ConnectionError('动作发送连接已断开') from self._action_reader_error
                self.robot_actions.send(self.msgpack.packb({'method':'execute_action','params':action}))
                sent=stamp()
            if self.recorder:self.recorder.write('initialization',{'status':'transport_sent','send_start_114':started,'send_return_114':sent,'ack_policy':'not_waited'})
            time.sleep(2)
        return True

    def close(self):
        self._action_reader_stop.set()
        for socket in (self.model,self.robot_actions,self.robot):
            if socket is not None:
                try:socket.close()
                except Exception:pass
        if self._action_reader is not None:self._action_reader.join(timeout=1)
