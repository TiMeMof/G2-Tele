"""Explicit end-of-task review. No annotations are submitted before Save."""
import json
from episode_control import Conflict, FAILURE_REASONS


def save_review(runtime, body):
    ep=runtime.episode
    if ep is None or body.get('episodeId')!=ep.id or ep.review_started_s is None:
        raise Conflict('请先点击失败，结束录制并打开接管段落')
    items=body.get('round_annotations')
    if not isinstance(items,list) or len(items)!=ep.rounds:
        raise ValueError('请逐一填写本次所有接管段落')
    annotations=[]
    for item in items:
        if not isinstance(item,dict) or type(item.get('round')) is not int:
            raise ValueError('无效接管段落')
        reasons=item.get('reasons',[]);note=item.get('note','')
        if not isinstance(reasons,list) or any(r not in FAILURE_REASONS for r in reasons):
            raise ValueError('无效接管原因')
        if not isinstance(note,str) or len(note)>2000:
            raise ValueError('每段说明最长2000字')
        if not (reasons or note.strip()) or ('other' in reasons and not note.strip()):
            raise ValueError('请为每段选择原因或填写说明，选择其他时需填写说明')
        annotations.append({'round':item['round'],'reasons':list(dict.fromkeys(reasons)),'note':note.strip()})
    if sorted(a['round'] for a in annotations)!=list(range(1,ep.rounds+1)):
        raise ValueError('接管段落存在重复或遗漏')
    annotations.sort(key=lambda a:a['round'])
    reasons=body.get('reasons',[]);note=body.get('note','')
    if not isinstance(reasons,list) or any(r not in FAILURE_REASONS for r in reasons):
        raise ValueError('无效整条失败原因')
    if not isinstance(note,str) or len(note)>2000:
        raise ValueError('整条说明最长2000字')
    reasons=list(dict.fromkeys(reasons or [r for a in annotations for r in a['reasons']]))
    if not note.strip() and annotations:
        note='；'.join(f"第{a['round']}段：{a['note']}" for a in annotations if a['note'])[:2000]
    if not reasons and note.strip():reasons=['other']
    if not reasons or ('other' in reasons and not note.strip()):
        raise ValueError('未发生接管时，请填写整条失败原因或说明')
    payload={'round_annotations':annotations,'reasons':reasons,'note':note.strip()}
    signature=json.dumps(payload,sort_keys=True,ensure_ascii=False)
    if ep.review_payload is not None and ep.review_payload!=signature:
        raise Conflict('上次保存已开始，请先重试完成原提交；完成后可在数据页查看')
    if ep.review_saved:
        return {'success':True,'episode':ep.snapshot(),'review':ep.review_detail}
    if not runtime.review_ready():
        raise Conflict('正在结束录制或等待旧请求返回，请稍后保存')
    ep.review_payload=signature
    recorder=ep.recording
    if ep.result is None:
        ep.command('failure',ep.id,ep.revision,'review-result-'+ep.id,reasons,note.strip())
    if not ep.review_result_queued:
        previous=recorder.seq
        try:recorder.write('result',{'result':ep.result,'command':'failure'})
        finally:ep.review_result_queued=recorder.seq>previous
    recorder.flush_events()
    if recorder.uploader.status()['pending_records']:
        raise OSError('模型记录仍在补传，草稿保留，请稍后重试保存')
    recorder.stop_collection(refresh_barrier=True)
    transport=recorder.uploader.transport
    detail=transport.request('GET','/api/records/'+ep.id)['episode']
    for annotation in annotations:
        current=next((r.get('annotation') for r in detail['rounds'] if r['round']==annotation['round']),None)
        if current and current.get('reasons')==annotation['reasons'] and current.get('note')==annotation['note']:
            continue
        detail=transport.request('POST','/api/records/'+ep.id+'/annotations',
            {**annotation,'revision':detail['session_revision']})['episode']
    if detail.get('archive_state')!='complete':
        detail=transport.request('POST','/api/records/'+ep.id+'/finalize',
            {'revision':detail['session_revision']})['episode']
    ep.review_saved=True
    ep.review_detail=detail
    return {'success':True,'episode':ep.snapshot(),'review':detail}
