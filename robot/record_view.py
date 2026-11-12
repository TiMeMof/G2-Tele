"""Read-only episode browser over the durable 114 copy; never imports robot APIs."""
from collections import OrderedDict
from datetime import datetime, timezone
import copy
import json
import math
from pathlib import Path
import re
import threading

ID = re.compile(r'^[0-9a-f]{32}$')
MAX_FILE = 1_000_000
MAX_EPISODE_BYTES = 64_000_000
MAX_RECORDS = 20_000


def number(value, default=None):
    return value if type(value) in (int, float) and math.isfinite(value) else default


def ns(value):
    try:
        return int(value) if isinstance(value, (str, int)) and not isinstance(value, bool) else None
    except ValueError:
        return None


def build_episode(episode_id, records, receipts=0, warnings=None):
    warnings = list(warnings or [])
    records = sorted(records, key=lambda r: (r.get('sequence', 0), r['record_id']))
    metadata = next((r for r in records if r['kind'] == 'metadata'), None)
    if metadata is None:
        warnings.append('缺少metadata记录')
    meta = metadata['data'] if metadata else {}
    events_by_seq = {}
    for r in records:
        if r['kind'] == 'events':
            for event in r['data'].get('events', []):
                if isinstance(event, dict) and isinstance(event.get('seq'), int) and number(event.get('elapsed_s')) is not None:
                    events_by_seq[event['seq']] = event
    events = sorted(events_by_seq.values(), key=lambda e: e['seq'])
    if events and sorted(events_by_seq) != list(range(1, max(events_by_seq)+1)):
        warnings.append('事件序号存在缺口；区间仅按已记录事件重建')
    created = next((e for e in events if e['kind'] == 'created'), None)
    base_stamp = (metadata or (records[0] if records else {})).get('recorded_114', {})
    origin_mono = ns(created.get('monotonic_ns')) if created else ns(base_stamp.get('monotonic_ns'))
    origin_wall = ns(created.get('wall_ns')) if created else ns(base_stamp.get('wall_ns'))
    if created:
        delta = round(created['elapsed_s']*1e9)
        if origin_mono is not None: origin_mono -= delta
        if origin_wall is not None: origin_wall -= delta
    boot_id = base_stamp.get('boot_id')

    def elapsed(stamp):
        if not isinstance(stamp, dict): return None
        value = ns(stamp.get('monotonic_ns'))
        if value is not None and origin_mono is not None and stamp.get('boot_id') == boot_id:
            return (value-origin_mono)/1e9
        # Do not silently join different monotonic clock domains or use a changed wall clock.
        return None

    result = None
    for r in records:
        if r['kind'] in ('result', 'session_end') and isinstance(r['data'].get('result'), dict):
            result = r['data']['result']
    if result is None:
        result = next((e['result'] for e in reversed(events) if isinstance(e.get('result'), dict)), None)
    ends = [r for r in records if r['kind'] == 'session_end']
    elapsed_records = [elapsed(r.get('recorded_114')) for r in records]
    duration = max([0.0]+[max(0,e['elapsed_s']) for e in events]+[t for t in elapsed_records if t is not None and t >= 0])
    if result: duration = max(duration, number(result.get('duration_s'), 0))
    review_start=next((e['elapsed_s'] for e in events if e['kind']=='review_started'),None)
    if review_start is not None:duration=min(duration,max(0,review_start))
    outcome = ('success' if result.get('success') is True else 'failure') if result else ('incomplete' if ends else 'ongoing')

    segments, rounds = [], []
    phase, since, round_id = 'initializing', 0.0, None
    active_round = None
    phase_labels = {'initializing':'初始化', 'model':'模型推理', 'human':'人工纠正', 'paused':'暂停 / 等待', 'ended':'结束后收尾', 'unknown':'记录缺口'}
    for event in sorted(events, key=lambda e:(e['elapsed_s'],e['seq'])):
        t = max(0,min(duration,event['elapsed_s']))
        k = event['kind']; next_phase = phase
        if k in ('model_started','model_resumed'):
            next_phase='model'
        elif k=='human_round_started':
            next_phase='human';round_id=event.get('round')
            active_round={'round':round_id,'start_s':t,'end_s':duration,'open':True,'limit_exceeded':event.get('limit_exceeded',False)}
            rounds.append(active_round)
        elif k=='paused' and phase!='human':next_phase='paused'
        elif k=='result_required' and phase!='human':next_phase='paused'
        elif k in {'result_decided','review_started'}:next_phase='ended'
        if next_phase!=phase:
            if t>since:segments.append({'start_s':since,'end_s':t,'kind':phase,'label':phase_labels[phase],'round':round_id if phase=='human' else None})
            if phase=='human' and active_round:
                active_round.update(end_s=t,open=False);active_round=None
            phase,since=next_phase,t
    if duration>since:segments.append({'start_s':since,'end_s':duration,'kind':phase,'label':phase_labels[phase],'round':round_id if phase=='human' else None})
    if ends and active_round: active_round['open']=False; active_round['truncated']=True
    for item in rounds:item['window_start_s']=max(0,item['start_s']-10)

    windows = sorted([{'start_s':r['window_start_s'],'end_s':r['end_s']} for r in rounds] + ([{'start_s':max(0,result['duration_s']-10),'end_s':result['duration_s']}] if result and number(result.get('duration_s')) is not None else []), key=lambda w:w['start_s'])
    merged_windows=[]
    for window in windows:
        if merged_windows and window['start_s'] <= merged_windows[-1]['end_s']:
            merged_windows[-1]['end_s']=max(merged_windows[-1]['end_s'],window['end_s'])
        else:merged_windows.append(dict(window))

    # These are runtime-observed edges, not the full native VR stream.
    hands={}
    for side in ('left','right'):
        state='unknown';at=0.0;track=[]
        for event in sorted(events,key=lambda e:(e['elapsed_s'],e['seq'])):
            value=None
            if event['kind']=='vr_edge' and event.get('side')==side and type(event.get('active')) is bool:
                value='active' if event['active'] else 'idle'
            elif event['kind']=='paused' and event.get('reason') in ('vr_unknown','vr_observer_restarted'):
                value='unknown'
            elif event['kind'] in ('model_started','model_resumed'):
                value='idle'  # runtime gates these transitions on fresh, released VR
            if value is None or value==state:continue
            t=max(0,min(duration,event['elapsed_s']))
            if t>at:track.append({'start_s':at,'end_s':t,'kind':state})
            state,at=value,t
        if duration>at:track.append({'start_s':at,'end_s':duration,'kind':state})
        hands[side]=track

    dispatches={}
    for r in records:
        if r['kind']=='dispatch' and isinstance(r['data'].get('chunk_id'),str):
            dispatches.setdefault(r['data']['chunk_id'],[]).append(r)
    chunks=[]
    for r in records:
        if r['kind']!='chunk':continue
        d=r['data']; cid=d.get('chunk_id')
        if not isinstance(cid,str) or not ID.fullmatch(cid):
            warnings.append('无效动作块ID');continue
        history=dispatches.get(cid,[])
        statuses=[x['data'].get('status') for x in history]
        status=statuses[-1] if statuses else 'predicted_only'
        sent=any(x in ('transport_sent','rpc_returned','ack_unknown') for x in statuses)
        starts=[x['data'].get('send_start_114') for x in history if x['data'].get('send_start_114')]
        start=elapsed(d.get('inference_start_114'));end=elapsed(d.get('inference_end_114'))
        chunks.append({'chunk_id':cid,'sequence':r['sequence'],'start_s':start,'end_s':end,'send_s':elapsed(starts[0]) if starts else None,
                       'shape':d.get('shape'),'status':status,'sent':sent,
                       'inference_ms':max(0,(end-start)*1000) if start is not None and end is not None else None,
                       'source_timestamp':d.get('observation',{}).get('timestamps_raw',{}).get('states')})
    chunks.sort(key=lambda c:(c['start_s'] if c['start_s'] is not None else math.inf,c['sequence']))
    anchors=[r.get('clock_anchor_at_recording') for r in records if isinstance(r.get('clock_anchor_at_recording'),dict)]
    uncertainties=[ns(a.get('uncertainty_ns')) for a in anchors]
    if any(a.get('clock_step_detected') for a in anchors):warnings.append('检测到墙钟跳变；禁止直接按墙钟精确叠加')
    if any(c['start_s'] is None for c in chunks):warnings.append('部分动作块的单调时钟域不匹配，未推测其时间位置')
    if not ends:warnings.append('记录尚未结束或缺少结束标记')
    if not result:warnings.append('尚无人工结果')
    wall_ms=origin_wall//1_000_000 if origin_wall is not None else None
    summary={'episode_id':episode_id,'task_id':meta.get('task_id','未知任务'),'started_wall_ns':str(origin_wall) if origin_wall is not None else None,
             'started_iso':datetime.fromtimestamp(wall_ms/1000,tz=timezone.utc).isoformat() if wall_ms is not None else None,
             'duration_s':duration,'outcome':outcome,'interventions':result.get('interventions',len(rounds)) if result else len(rounds),
             'chunk_count':len(chunks),'record_count':len(records),'acknowledged_count':receipts,'pending_count':max(0,len(records)-receipts),
             'complete':bool(result and ends and not any(any(term in w for term in ('缺口','无效','缺少','损坏','符号链接')) for w in warnings)),
             'directory_226':str((metadata or {}).get('data_dir',meta.get('config',{}).get('data_dir','/data/yfmo'))).rstrip('/')+'/action_chunks/'+episode_id,
             'reasons':result.get('reasons',[]) if result else [],'note':result.get('note','') if result else '',
             'model_independent_success':result.get('model_independent_success') if result else None,'warnings':list(dict.fromkeys(warnings))}
    detail={**summary,'metadata':meta,'result':result,'review_windows_merged':merged_windows,'segments':segments,'rounds':rounds,'hands':hands,'events':events,'chunks':chunks,
            'time_basis':'114 monotonic relative to episode start','hand_track_basis':'runtime_detected_edges_not_full_vr',
            'clock':{'boot_id_114':boot_id,'origin_monotonic_ns':str(origin_mono) if origin_mono is not None else None,
                     'max_roundtrip_uncertainty_ns':str(max(v for v in uncertainties if v is not None)) if any(v is not None for v in uncertainties) else None},
            'feedback':{'available':False,'reason':'缺少与本次任务对应时段的数采数据；不使用历史诊断样本替代实际反馈。'},
            'source':'114 durable mirror of records sent to 226; acknowledgements shown separately'}
    return summary,detail


class RecordBrowser:
    def __init__(self, root):
        self.root=Path(root).resolve();self.lock=threading.RLock();self.cache=OrderedDict()

    def folder(self, episode_id):
        if not isinstance(episode_id,str) or not ID.fullmatch(episode_id):raise ValueError('无效任务记录ID')
        path=self.root/episode_id
        if path.is_symlink() or path.resolve().parent!=self.root:raise ValueError('路径不允许')
        if not path.is_dir():raise FileNotFoundError('记录不存在')
        return path

    def _load(self, episode_id):
        folder=self.folder(episode_id)
        entries=[];signature=[];size=0;receipts=set();warnings=[]
        for p in folder.iterdir():
            if p.suffix!='.json':continue
            if p.is_symlink():warnings.append('已忽略符号链接');continue
            if p.name.endswith('.receipt.json'):
                if ID.fullmatch(p.name[:-13]):receipts.add(p.name[:-13])
                signature.append((p.name,p.stat().st_size,p.stat().st_mtime_ns));continue
            if not ID.fullmatch(p.stem):continue
            st=p.stat();size+=st.st_size;entries.append(p);signature.append((p.name,st.st_size,st.st_mtime_ns))
            if len(entries)>MAX_RECORDS or size>MAX_EPISODE_BYTES:raise ValueError('记录过大，超出当前浏览上限')
        signature=tuple(sorted(signature))
        with self.lock:
            cached=self.cache.get(episode_id)
            if cached and cached[0]==signature:
                self.cache.move_to_end(episode_id);return cached[1:]
            records=[]
            for p in entries:
                try:
                    if p.stat().st_size>MAX_FILE:raise ValueError('file limit')
                    r=json.loads(p.read_text(encoding='utf-8'),parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite')))
                    if r.get('schema')!='action-chunk-events/1' or r.get('episode_id')!=episode_id or r.get('record_id')!=p.stem or not isinstance(r.get('data'),dict) or type(r.get('sequence')) is not int:raise ValueError('schema mismatch')
                    records.append(r)
                except (OSError,ValueError,TypeError,AttributeError):warnings.append('部分记录损坏或格式无效，已跳过')
            ack=sum(1 for r in records if r['record_id'] in receipts)
            summary,detail=build_episode(episode_id,records,ack,warnings)
            self.cache[episode_id]=(signature,summary,detail,records)
            while len(self.cache)>8:self.cache.popitem(last=False)
            return summary,detail,records

    def listing(self, params):
        def query(key):return params.get(key,[''])[0].strip()
        limit=max(1,min(100,int(query('limit') or 50)));offset=max(0,int(query('offset') or 0))
        items=[];errors=[]
        folders=[] if not self.root.exists() else [p for p in self.root.iterdir() if ID.fullmatch(p.name) and p.is_dir() and not p.is_symlink()]
        for folder in folders:
            try:
                summary,_,_=self._load(folder.name)
                if query('task') and query('task').casefold() not in summary['task_id'].casefold():continue
                if query('outcome') and query('outcome')!=summary['outcome']:continue
                if query('human')=='yes' and not summary['interventions']:continue
                if query('human')=='no' and summary['interventions']:continue
                if query('date') and summary['started_wall_ns']:
                    # Website uses local Asia/Shanghai calendar dates, explicitly matched here.
                    from datetime import timedelta
                    date=datetime.fromtimestamp(int(summary['started_wall_ns'])/1e9,tz=timezone(timedelta(hours=8))).date().isoformat()
                    if date!=query('date'):continue
                items.append(summary)
            except (OSError,ValueError,TypeError,KeyError) as exc:errors.append({'episode_id':folder.name,'message':str(exc)[:160]})
        items.sort(key=lambda x:int(x['started_wall_ns'] or 0),reverse=True)
        return {'success':True,'records':copy.deepcopy(items[offset:offset+limit]),'total':len(items),'offset':offset,'limit':limit,'errors':errors}

    def detail(self, episode_id):return {'success':True,'episode':copy.deepcopy(self._load(episode_id)[1])}

    def chunk(self, episode_id, chunk_id):
        if not ID.fullmatch(chunk_id):raise ValueError('无效动作块ID')
        records=self._load(episode_id)[2]
        record=next((r for r in records if r['kind']=='chunk' and r['data'].get('chunk_id')==chunk_id),None)
        if record is None:raise FileNotFoundError('动作块不存在')
        return {'success':True,'record':copy.deepcopy(record),'dispatches':copy.deepcopy([r for r in records if r['kind']=='dispatch' and r['data'].get('chunk_id')==chunk_id])}
