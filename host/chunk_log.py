"""Durable 114 spool and authenticated 226 transfer with explicit clock anchors."""
import copy
import hashlib
import http.client
import json
import os
from pathlib import Path
import queue
import socket
import tempfile
import threading
import time
import uuid
from network_config import get_robot_host, load

BASE=Path(__file__).resolve().parent
BOOT=Path('/proc/sys/kernel/random/boot_id').read_text().strip() if Path('/proc/sys/kernel/random/boot_id').exists() else 'test'
def stamp():return {'host':socket.gethostname(),'wall_ns':str(time.time_ns()),'monotonic_ns':str(time.monotonic_ns()),'boot_id':BOOT}
def encode(data):return json.dumps(data,ensure_ascii=False,allow_nan=False,sort_keys=True,separators=(',',':')).encode('utf-8')
def atomic(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.pending-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        os.replace(name,path)
        if hasattr(os,'O_DIRECTORY'):
            fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
    finally:
        if os.path.exists(name):os.unlink(name)

class Transport:
    def __init__(self,host=None,port=None,token_file=None):
        self._host,self.port=host,port or load()["collection_port"];self.token_file=Path(token_file or BASE/'session_receiver.token')
    @property
    def host(self):
        return self._host or get_robot_host()
    def request(self,method,path,body=None):
        timeout=1 if path=="/api/health" else 3 if method=="GET" and path.startswith("/api/records?") else 20
        conn=http.client.HTTPConnection(self.host,self.port,timeout=timeout)
        try:
            token=self.token_file.read_text().strip()
            conn.request(method,path,body=None if body is None else encode(body),headers={'Authorization':'Bearer '+token,'Content-Type':'application/json','Connection':'close'})
            response=conn.getresponse();raw=response.read(16000001)
            if len(raw)>16000000:raise OSError('机器人记录响应过大')
            if response.status!=200:
                try: message=json.loads(raw).get('message')
                except ValueError: message=None
                raise OSError(message or '机器人记录服务 HTTP '+str(response.status))
            data=json.loads(raw)
            if data.get('success') is not True:raise OSError(data.get('message') or '机器人记录被拒绝')
            return data
        finally:conn.close()
    def anchor(self):
        begin=stamp();data=self.request('GET','/api/clock');end=stamp()
        a,b=int(begin['wall_ns']),int(end['wall_ns']);c,d=int(data['received']['wall_ns']),int(data['sent']['wall_ns'])
        rtt=int(end['monotonic_ns'])-int(begin['monotonic_ns'])
        service=int(data['sent']['monotonic_ns'])-int(data['received']['monotonic_ns'])
        stable=abs((b-a)-rtt)<5000000 and abs((d-c)-service)<5000000
        return {'client_send':begin,'server_receive':data['received'],'server_send':data['sent'],'client_receive':end,
                'offset_226_minus_114_ns':str(((c-a)+(d-b))//2),'uncertainty_ns':str(max(0,rtt-service)//2),
                'roundtrip_ns':str(rtt),'clock_step_detected':not stable,'method':'four_timestamp_roundtrip','not_hardware_synchronization':True}

class Uploader:
    def __init__(self,spool=None,transport=None):
        self.root=Path(spool or BASE/'pending_spool');self.transport=transport or Transport(host=get_robot_host())
        self.queue=queue.Queue();self.lock=threading.Lock();self.pending=set();self.stop=threading.Event();self.thread=None
        self.last_error=None;self.anchor_value=None;self.anchor_at=0
    def start(self):
        with self.lock:
            if self.thread:return
            self.root.mkdir(parents=True,exist_ok=True)
            for folder in self.root.iterdir():
                if not folder.is_dir():continue
                files=[p for p in folder.glob('*.json') if not p.name.endswith('.receipt.json')]
                records=[json.loads(p.read_text(encoding='utf-8')) for p in files]
                if records and any(r['kind']=='metadata' for r in records) and not any(r['kind'] in {'result','session_end'} for r in records):
                    first=records[0];rid=uuid.uuid4().hex
                    recovered={'schema':'action-chunk-events/1','record_id':rid,'episode_id':first['episode_id'],
                        'sequence':max(r.get('sequence',0) for r in records)+1,'data_dir':first['data_dir'],
                        'kind':'session_end','recorded_114':stamp(),'data':{'phase':'interrupted_by_web_restart','result':None,'auto_resumed':False}}
                    path=folder/(rid+'.json');atomic(path,encode(recovered));files.append(path)
                for path in files:
                    if not path.with_suffix('.receipt.json').exists():self.enqueue(path)
            self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
    def enqueue(self,path):
        key=str(path)
        if key not in self.pending:self.pending.add(key);self.queue.put(path)
    def put(self,path):
        if self.anchor_value is None or time.monotonic()-self.anchor_at>5:
            self.anchor_value=self.transport.anchor();self.anchor_at=time.monotonic()
        record=json.loads(path.read_text(encoding='utf-8'))
        reply=self.transport.request('POST','/api/records',{'record':record,'clock_anchor':self.anchor_value})
        digest=hashlib.sha256(encode(record)).hexdigest()
        if reply.get('record_sha256')!=digest:raise OSError('226 hash mismatch')
        path.unlink()  # acknowledged payload lives only on the robot; host cache is temporary
    def run(self):
        while not self.stop.is_set():
            try:path=self.queue.get(timeout=.2)
            except queue.Empty:continue
            try:
                if not path.with_suffix('.receipt.json').exists():self.put(path)
                self.pending.discard(str(path));self.last_error=None
            except Exception as e:
                self.last_error=type(e).__name__+': '+str(e)[:200]
                self.queue.put(path);self.stop.wait(1)
            finally:self.queue.task_done()
    def status(self):return {'pending_records':len(self.pending),'transfer_error':self.last_error,'local_spool':str(self.root)}

class Recorder:
    def __init__(self,episode,uploader):
        self.episode=episode;self.uploader=uploader;self.lock=threading.RLock();self.event_seq=0;self.seq=0
        self.folder=uploader.root/episode.id;self.directory=episode.config['data_dir'];self.last_chunk_id=None
        self.collection_stopped = False
        reply = uploader.transport.request('POST','/api/sessions', {
            'episode_id':episode.id, 'task_id':episode.task_id,
            'category':episode.config.get('task_category','洗衣机'), 'policy':episode.policy, 'data_dir':episode.config['data_dir']})
        self.session = reply['session']
        duration_ms = self.session.get('collector',{}).get('max_duration_ms')
        if type(duration_ms) is int:
            episode.collection_deadline = episode.started + duration_ms/1000 - 5
    def write(self,kind,data):
        with self.lock:
            self.seq+=1;rid=uuid.uuid4().hex
            record={'schema':'action-chunk-events/1','record_id':rid,'episode_id':self.episode.id,'sequence':self.seq,
                    'data_dir':self.directory,'kind':kind,'recorded_114':stamp(),'data':data,
                    'clock_anchor_at_recording':copy.deepcopy(self.uploader.anchor_value),
                    'clock_anchor_age_ms':(time.monotonic()-self.uploader.anchor_at)*1000 if self.uploader.anchor_value else None}
            raw=encode(record)
            if len(raw)>1000000:raise ValueError('record too large')
            path=self.folder/(rid+'.json');atomic(path,raw)
            try:
                self.uploader.put(path)  # robot acknowledgement before subsequent action sends
            except Exception:
                self.uploader.enqueue(path)
                raise
            return rid
    def flush_events(self):
        snapshot=self.episode.snapshot()
        with self.lock:
            new=[e for e in snapshot['events'] if e['seq']>self.event_seq]
            if new:
                if new[0]['seq']!=self.event_seq+1:raise OSError('episode event log gap')
                self.write('events',{'events':new,'phase':snapshot['phase']});self.event_seq=new[-1]['seq']
    def stop_collection(self, refresh_barrier=False):
        if not self.collection_stopped or refresh_barrier:
            self.session = self.uploader.transport.request('POST','/api/sessions/'+self.episode.id+'/stop',{'expected_sequence':self.seq})['session']
            self.collection_stopped = True

    def status(self):
        return {'enabled':True,'directory_226':self.session.get('raw_directory','').rsplit('/raw',1)[0],
                'collection_stopped':self.collection_stopped, **self.uploader.status()}

_uploader=None
_lock=threading.Lock()
def shared_uploader():
    global _uploader
    with _lock:
        if _uploader is None:_uploader=Uploader();_uploader.start()
        return _uploader
