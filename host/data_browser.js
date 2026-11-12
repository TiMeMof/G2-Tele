'use strict';
const DataBrowser = (() => {
    const $ = id => document.getElementById(id);
    const outcomeNames={success:'成功',failure:'失败',ongoing:'未结束',incomplete:'异常结束 / 未标注'};
    const statusNames={predicted_only:'仅有预测',intent_not_sent:'准备发送，未确认发送',transport_sent:'已发送，等待回复',rpc_returned:'已发送，接口已返回',ack_unknown:'已发送，确认未知',discarded_before_dispatch:'接管后丢弃',discarded_before_send:'发送前丢弃',dispatch_error:'发送异常',diagnostic_not_sent:'诊断，未发送'};
    const eventNames={created:'任务创建',model_started:'模型开始',model_resumed:'继续推理',human_round_started:'人工纠正开始',paused:'暂停',vr_edge:'VR按键变化',result_required:'等待人工结果',result_decided:'人工结果',dispatch_started:'准备动作',transport_sent:'动作发送',dispatch_returned:'接口返回',dispatch_discarded:'发送已丢弃',inference_discarded:'推理已丢弃',reset_returned:'复位返回'};
    const reasonNames={vr_takeover:'VR接管',manual_pause:'网页中止',vr_unknown:'VR状态未知',vr_observer_restarted:'VR观察服务重启',intervention_limit:'接管次数用尽',runtime_error:'执行异常',model_finished:'模型结束',wrong_pose:'目标位置或姿态错误',manual_reset:'人工重置',initial_out_of_distribution:'初始状态不适合模型',repetitive_actions:'相似动作循环',no_progress:'推进停滞',collision:'碰撞',preventive_stop:'预防性终止',grasp_failed:'抓取失败',object_slipped:'物体滑落',wrong_stage:'动作阶段错误',unreachable:'限位或不可达',unstable_actions:'抖动或发散',other:'其他'};
    let initialized=false,listEpoch=0,detailEpoch=0,chunkEpoch=0,timer,rows=[],total=0,currentId=null,detail=null,chunk=null,selectedChunk=null,selectedRound=null;
    let view=[0,1];
    let annotationQueue=Promise.resolve();
    const annotationTimers=new Map();
    const annotationDrafts=new Map();
    const fmt=v=>Number.isFinite(v)?v.toFixed(2)+'s':'未知';
    function date(value) {try{return new Date(Number(BigInt(value)/1000000n)).toLocaleString('zh-CN',{timeZone:'Asia/Shanghai',hour12:false});}catch{return '日期未知';}}
    function node(tag,text,cls){const el=document.createElement(tag);if(text!==undefined)el.textContent=text;if(cls)el.className=cls;return el;}
    async function get(url){try{const r=await fetch(url,{signal:AbortSignal.timeout(30000)});const d=await r.json();if(!r.ok||!d.success)throw new Error(d.message||'读取失败');return d;}catch(e){if(e.name==='TimeoutError')throw new Error('读取超过30秒，请刷新重试；读取超时不表示采集数据为空。');throw e;}}
    function error(message){$('dataError').textContent=message||'';}
    function init(){if(initialized)return;initialized=true;
        const names=[...Array.from({length:7},(_,i)=>'左臂 J'+(i+1)),'左夹爪',...Array.from({length:7},(_,i)=>'右臂 J'+(i+1)),'右夹爪'];
        names.forEach((name,i)=>{const opt=node('option',name);opt.value=i;$('chunkChannel').append(opt);});
    }
    function filter(){clearTimeout(timer);timer=setTimeout(()=>refresh(),250);}
    async function refresh(append=false){init();const request=++listEpoch;error('');$('recordCount').textContent='正在读取记录…';
        const params=new URLSearchParams({task:$('dataTaskFilter').value.trim(),date:$('dataDateFilter').value,outcome:$('dataResultFilter').value,human:$('dataHumanFilter').value,limit:'50',offset:String(append?rows.length:0)});
        try {const data=await get('/api/records?'+params);if(request!==listEpoch)return;
            rows=append?[...rows,...data.records]:data.records;total=data.total;renderList();
            if(data.errors?.length)error(`${data.errors.length} 条记录无法读取：${data.errors[0].message}`);
            if(currentId && rows.some(r=>r.episode_id===currentId))await select(currentId,true);
            else if(rows.length)await select(rows[0].episode_id);
            else {currentId=null;detail=null;detailEpoch++;chunkEpoch++;$('recordDetail').hidden=true;$('recordEmpty').hidden=false;$('recordEmpty').textContent=data.errors?.length?'记录源暂时不可达，请检查连接后刷新。':'没有符合筛选条件的任务记录。';}
        } catch(e){if(request===listEpoch){error('读取记录失败：'+e.message);$('recordCount').textContent='读取失败，可点击刷新重试';}}
    }
    function renderList(){const list=$('recordList');list.replaceChildren();$('recordCount').textContent=`${total} 条记录 · 时间倒序`;$('recordMore').hidden=rows.length>=total;
        for(const item of rows){const b=node('button',undefined,'record-option');b.type='button';b.setAttribute('aria-current',String(item.episode_id===currentId));b.dataset.episodeId=item.episode_id;
            b.append(node('strong',item.task_id));b.append(node('span',date(item.started_wall_ns),'record-muted'));
            const line=node('span',undefined,'record-line');line.append(node('span',item.archive_state==='complete'?'已归档':item.collection_state==='stopped'?'待标注':outcomeNames[item.outcome]||item.outcome,'result-chip '+item.outcome));line.append(node('span',`${fmt(item.duration_s)} · 接管 ${item.interventions} 轮`));b.append(line);
            b.append(node('span',`${item.chunk_count} 个动作块 · ${item.pending_count?item.pending_count+'条待同步':'机器人已保存'}${item.complete?'':' · 记录未完整'}`,'record-muted'));
            b.onclick=()=>select(item.episode_id);list.append(b);
        }
    }
    async function select(id,keep=false){const request=++detailEpoch;++chunkEpoch;currentId=id;renderList();error('');
        $('recordDetail').hidden=true;$('recordEmpty').hidden=false;$('recordEmpty').textContent='正在读取任务时间轴…';
        try{const data=await get('/api/records/'+encodeURIComponent(id));if(request!==detailEpoch)return;
            const same=detail?.episode_id===id;detail=data.episode;chunk=null;
            if(!keep||!same){view=[0,Math.max(.1,detail.duration_s)];selectedChunk=null;selectedRound=null;}
            $('recordEmpty').hidden=true;$('recordDetail').hidden=false;renderDetail();
        }catch(e){if(request===detailEpoch){detail=null;$('recordEmpty').textContent='这条记录暂时无法读取。';error(e.message);}}
    }
    function ensureReviewPanel(){
        if($('reviewControls'))return;
        const wrap=node('section',undefined,'review-controls');wrap.id='reviewControls';
        const progress=node('p');progress.id='annotationProgress';wrap.append(progress);
        const finish=node('button','完成归档');finish.id='finalizeSession';finish.type='button';
        finish.onclick=()=>finalizeSession();wrap.append(finish);
        const stop=node('button','结束遗留录制 / 重试结束');stop.id='stopRawSession';stop.type='button';
        stop.onclick=()=>stopRawSession();wrap.append(stop);
        const panel=node('div',undefined,'round-annotation');panel.id='roundAnnotationPanel';
        panel.append(node('h4','接管原因'));
        const title=node('p');title.id='annotationTitle';panel.append(title);
        const choices=node('div',undefined,'annotation-choices');choices.id='annotationReasons';panel.append(choices);
        const label=node('label','补充说明（也可只填写具体原因）');label.htmlFor='annotationNote';panel.append(label);
        const note=node('textarea');note.id='annotationNote';note.maxLength=2000;note.rows=3;
        note.placeholder='这次为什么接管，做了什么纠正？';note.oninput=()=>scheduleAnnotation();panel.append(note);
        const save=node('button','保存本次接管标注');save.id='saveRoundAnnotation';save.type='button';
        save.onclick=()=>scheduleAnnotation(true);panel.append(save);
        const state=node('span');state.id='annotationSaveStatus';state.setAttribute('role','status');panel.append(state);
        wrap.append(panel);$('roundButtons').after(wrap);
    }
    async function post(url,body){
        const response=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
        const value=await response.json();
        if(!response.ok||!value.success)throw new Error(value.message||'保存失败');
        return value;
    }
    function renderAnnotation(){
        ensureReviewPanel();
        const legacy=!!detail?.legacy;
        $('reviewControls').hidden=legacy;
        if(legacy)return;
        const missing=detail.missing_annotations||[];
        $('annotationProgress').textContent=detail.archive_state==='complete'?'已完成归档':
            '待归档：'+missing.length+' 次接管尚未标注。录制状态：'+(detail.collection_state==='stopped'?'已结束':detail.collection_state||'未知');
        $('finalizeSession').disabled=!detail.can_finalize||detail.archive_state==='complete';
        $('finalizeSession').textContent=detail.archive_state==='complete'?'已归档':'完成归档';
        $('stopRawSession').hidden=!['recording','stop_failed','start_failed','preparing'].includes(detail.collection_state);
        const r=detail.rounds[selectedRound];
        $('roundAnnotationPanel').hidden=!r;
        if(!r)return;
        $('annotationTitle').textContent='第 '+r.round+' 次接管 · '+fmt(r.start_s)+' – '+fmt(r.end_s);
        const key=currentId+'/'+r.round;
        const value=annotationDrafts.get(key)||r.annotation||{reasons:[],note:''};
        const choices=$('annotationReasons');choices.replaceChildren();
        for(const [id,name] of Object.entries(detail.reason_options||{})){
            const label=node('label');const input=node('input');input.type='checkbox';input.value=id;
            input.checked=(value.reasons||[]).includes(id);input.disabled=detail.archive_state==='complete';
            input.onchange=()=>scheduleAnnotation();label.append(input,document.createTextNode(name));choices.append(label);
        }
        $('annotationNote').value=value.note||'';
        $('annotationNote').disabled=detail.archive_state==='complete';
        $('saveRoundAnnotation').disabled=detail.archive_state==='complete';
        $('annotationSaveStatus').textContent=annotationDrafts.has(key)?'草稿待保存':r.annotation?'已保存':'尚未标注';
    }
    function scheduleAnnotation(immediate=false){
        if(!detail||selectedRound===null)return;
        const r=detail.rounds[selectedRound];if(!r)return;
        const sid=currentId,key=sid+'/'+r.round;
        const value={round:r.round,reasons:[...$('annotationReasons').querySelectorAll('input:checked')].map(n=>n.value),note:$('annotationNote').value};
        annotationDrafts.set(key,value);$('annotationSaveStatus').textContent='草稿待保存';
        // Each queued value captures its own episode and round; changing selection cannot relabel another interval.
        if(annotationTimers.has(key))clearTimeout(annotationTimers.get(key).timer);
        const send=()=>{
            annotationTimers.delete(key);
            annotationQueue=annotationQueue.catch(()=>{}).then(async()=>{
                try{
                    const latest=await get('/api/records/'+sid);
                    const response=await post('/api/records/'+sid+'/annotations',{...value,revision:latest.episode.session_revision});
                    if(annotationDrafts.get(key)===value)annotationDrafts.delete(key);
                    if(currentId===sid){detail=response.episode;if(selectedRound!==null&&detail.rounds[selectedRound]?.round===value.round)renderAnnotation();
                        else $('annotationProgress').textContent='已保存第 '+value.round+' 次接管原因';
                        $('finalizeSession').disabled=!detail.can_finalize;}
                }catch(e){error('标注未保存：'+e.message);if(currentId===sid)$('annotationSaveStatus').textContent='保存失败，草稿仍在页面中';}
            });
        };
        if(immediate)send();else annotationTimers.set(key,{timer:setTimeout(send,500),flush:send});
    }
    async function finalizeSession(){
        try{
            for(const pending of [...annotationTimers.values()]){clearTimeout(pending.timer);pending.flush();}
            await annotationQueue;
            if([...annotationDrafts.keys()].some(k=>k.startsWith(currentId+'/')))throw new Error('请先保存所有接管标注');
            const latest=await get('/api/records/'+currentId);
            const response=await post('/api/records/'+currentId+'/finalize',{revision:latest.episode.session_revision});
            detail=response.episode;renderDetail();await refresh();
        }catch(e){error('归档未完成：'+e.message);}
    }
    async function stopRawSession(){
        try{await post('/api/records/'+currentId+'/stop',{});await select(currentId,true);}
        catch(e){error('录制未确认结束：'+e.message);}
    }
    function renderDetail(){if(!detail)return;const d=detail;
        $('recordTitle').textContent=(d.category?d.category+' / ':'')+d.task_id;$('recordDate').textContent=date(d.started_wall_ns)+' · '+d.episode_id;
        $('recordOutcome').textContent=outcomeNames[d.outcome]||d.outcome;$('recordOutcome').className='result-chip '+d.outcome;
        $('recordDuration').textContent=fmt(d.duration_s);$('recordRounds').textContent=d.interventions+' 轮';$('recordChunks').textContent=d.chunk_count;$('recordPending').textContent=d.pending_count+' 条';
        const reasons=(d.reasons||[]).map(r=>reasonNames[r]||r).join('、');
        $('recordResult').textContent=(reasons?'失败原因：'+reasons+'。':'')+(d.note?'说明：'+d.note+'。':'')+(d.outcome==='success'?(d.model_independent_success?'模型独立完成。':'经人工纠正后完成。'):'');
        $('recordWarnings').hidden=!d.warnings?.length;$('recordWarnings').textContent=(d.warnings||[]).join('；');
        $('feedbackGap').textContent=d.raw_directory?'原始关节、VR和相机数据保存在机器人；当前图显示模型动作，原始反馈曲线尚未载入。':'实际关节 / 完整VR曲线：'+d.feedback.reason;
        const uncertainty=d.clock?.max_roundtrip_uncertainty_ns;
        $('recordStorage').textContent=`机器人目录：${d.directory_226}。${d.acknowledged_count}/${d.record_count} 条记录已确认落盘。时间轴使用113单调时钟。${uncertainty?'跨机往返测量最大不确定性估计 ±'+(Number(uncertainty)/1e6).toFixed(2)+'ms，不等于硬件同步精度。':''}`;
        const buttons=$('roundButtons');buttons.replaceChildren();
        if(!d.rounds.length)buttons.append(node('span','本次未记录人工接管轮次','help'));
        d.rounds.forEach((r,i)=>{const b=node('button',`第${r.round}轮 · ${fmt(r.start_s)}`);b.dataset.round=i;b.textContent+=' · '+(r.annotation?'已标注':'待标注');b.onclick=()=>round(i);buttons.append(b);});
        renderWindow();
    }
    function clipped(s){const begin=Math.max(view[0],s.start_s),end=Math.min(view[1],s.end_s);return end>begin?[begin,end]:null;}
    function renderTrack(id,segments,isPhase=false){const track=$(id);track.replaceChildren();const width=view[1]-view[0];
        segments.forEach(s=>{const area=clipped(s);if(!area)return;const button=node('button',isPhase&&area[1]-area[0]>width*.07?(s.kind==='human'?'人工 '+s.round:s.label):'', 'timeline-segment '+s.kind);
            button.style.left=((area[0]-view[0])/width*100)+'%';button.style.width=((area[1]-area[0])/width*100)+'%';
            const label=(s.label||{active:'接管按下',idle:'松开',unknown:'未知'}[s.kind]||s.kind)+`：${fmt(s.start_s)}–${fmt(s.end_s)}`;
            button.title=label;button.setAttribute('aria-label',label);
            button.onclick=()=>{if(isPhase&&s.kind==='human'){const i=detail.rounds.findIndex(r=>r.round===s.round);if(i>=0)round(i);}else{view=[Math.max(0,s.start_s-10),Math.max(s.start_s+.1,s.end_s)];selectedRound=null;renderWindow();}};
            track.append(button);
        });
    }
    function renderWindow(){if(!detail)return;const duration=Math.max(.1,detail.duration_s);view[0]=Math.max(0,Math.min(view[0],duration-.1));view[1]=Math.max(view[0]+.1,Math.min(view[1],duration));
        $('timelineWindow').textContent=fmt(view[0])+' – '+fmt(view[1]);renderTrack('phaseTrack',detail.segments,true);renderTrack('leftTrack',detail.hands.left);renderTrack('rightTrack',detail.hands.right);
        $('timelineTicks').replaceChildren(...Array.from({length:5},(_,i)=>node('span',fmt(view[0]+(view[1]-view[0])*i/4))));
        const range=duration-(view[1]-view[0]);$('timelinePan').disabled=range<.01;$('timelinePan').value=range>0?Math.round(view[0]/range*1000):0;
        $('previousRound').disabled=!detail.rounds.length||selectedRound===0;$('nextRound').disabled=!detail.rounds.length||selectedRound===detail.rounds.length-1;
        document.querySelectorAll('#roundButtons button').forEach((b,i)=>b.setAttribute('aria-pressed',String(i===selectedRound)));
        renderChunks();renderEvents();renderAnnotation();
    }
    function round(i){if(!detail?.rounds[i])return;selectedRound=i;const r=detail.rounds[i];view=[r.window_start_s,Math.max(r.start_s+.1,r.end_s)];renderWindow();}
    function stepRound(direction){if(!detail)return;const i=selectedRound===null?(direction>0?0:detail.rounds.length-1):selectedRound+direction;round(i);}
    function full(){if(!detail)return;selectedRound=null;view=[0,Math.max(.1,detail.duration_s)];renderWindow();}
    function ending(){if(!detail)return;const end=detail.result?.duration_s??detail.duration_s;selectedRound=null;view=[Math.max(0,end-10),Math.max(.1,end)];renderWindow();}
    function zoom(factor){if(!detail)return;const duration=Math.max(.1,detail.duration_s),width=Math.min(duration,Math.max(.5,(view[1]-view[0])*factor)),mid=(view[0]+view[1])/2;
        const start=Math.max(0,Math.min(duration-width,mid-width/2));view=[start,start+width];selectedRound=null;renderWindow();}
    function pan(value){if(!detail)return;const width=view[1]-view[0],start=(Math.max(.1,detail.duration_s)-width)*Number(value)/1000;view=[start,start+width];selectedRound=null;renderWindow();}
    function renderChunks(){const list=$('chunkList');list.replaceChildren();
        const items=detail.chunks.filter(c=>c.start_s!==null && c.start_s<=view[1] && (c.end_s??c.start_s)>=view[0]);
        $('chunkWindowCount').textContent=`当前窗口 ${items.length} / 全程 ${detail.chunks.length} 个动作块`;
        if(!items.length){++chunkEpoch;chunk=null;selectedChunk=null;list.append(node('p','该时段没有模型动作块','help'));drawChunk();return;}
        if(!items.some(c=>c.chunk_id===selectedChunk))selectedChunk=items[0].chunk_id;
        items.forEach((c,i)=>{const b=node('button',`${fmt(c.start_s)} · ${c.shape?.join('×')||'未知形状'}\n${statusNames[c.status]||c.status}`,'chunk-option');b.style.whiteSpace='pre-line';b.dataset.chunkId=c.chunk_id;b.setAttribute('aria-current',String(c.chunk_id===selectedChunk));b.onclick=()=>chooseChunk(c.chunk_id);list.append(b);});
        if(chunk?.record.data.chunk_id!==selectedChunk)chooseChunk(selectedChunk);else drawChunk();
    }
    async function chooseChunk(id){const request=++chunkEpoch,episodeId=currentId;selectedChunk=id;chunk=null;$('chunkInfo').textContent='正在读取动作块…';$('chunkChart').replaceChildren();
        document.querySelectorAll('#chunkList button').forEach(b=>b.setAttribute('aria-current',String(b.dataset.chunkId===id)));
        try{const data=await get('/api/records/'+episodeId+'/chunks/'+id);if(request!==chunkEpoch||episodeId!==currentId)return;chunk=data;drawChunk();}
        catch(e){if(request===chunkEpoch){$('chunkInfo').textContent='动作块读取失败：'+e.message;}}
    }
    function svg(tag,attrs,text){const el=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v] of Object.entries(attrs))el.setAttribute(k,String(v));if(text!==undefined)el.textContent=text;return el;}
    function drawChunk(){const target=$('chunkChart');target.replaceChildren();if(!chunk){target.append(svg('text',{x:290,y:120,'text-anchor':'middle'},'请选择窗口内的动作块'));$('chunkInfo').textContent='';return;}
        const data=chunk.record.data,index=Number($('chunkChannel').value),values=(data.actions||[]).map(row=>row[index]);
        const command=chunk.dispatches.find(r=>r.data.command)?.data.command;
        let prepared=[];if(command){if(index<7)prepared=(command.left_arm?.values||[]).map(row=>row[index]);else if(index===7)prepared=(command.left_effector||[]).map(row=>row[0]);else if(index<15)prepared=(command.right_arm?.values||[]).map(row=>row[index-8]);else prepared=(command.right_effector||[]).map(row=>row[0]);}
        const snapshot=data.input_state?.[index];const numeric=[...values,...prepared,snapshot].filter(v=>typeof v==='number'&&Number.isFinite(v));
        if(!numeric.length){target.append(svg('text',{x:290,y:120,'text-anchor':'middle'},'该通道无有效数值'));return;}
        let low=Math.min(...numeric),high=Math.max(...numeric),pad=Math.max(.025,(high-low)*.12);low-=pad;high+=pad;
        const X=i=>55+i/Math.max(1,values.length-1)*505,Y=v=>205-(v-low)/(high-low)*175;
        for(let i=0;i<5;i++){const y=30+i*175/4;target.append(svg('line',{x1:55,x2:560,y1:y,y2:y,stroke:'#e2e8f0'}));target.append(svg('text',{x:48,y:y+4,'text-anchor':'end'},(high-i*(high-low)/4).toFixed(3)));}
        [0,Math.floor((values.length-1)/2),values.length-1].forEach(i=>target.append(svg('text',{x:X(i),y:226,'text-anchor':'middle'},String(i))));
        function path(series,color){let d='',drawing=false;series.forEach((v,i)=>{if(typeof v!=='number'||!Number.isFinite(v)){drawing=false;return;}d+=(drawing?' L ':' M ')+X(i)+' '+Y(v);drawing=true;});target.append(svg('path',{d,stroke:color,class:'plot-line'}));}
        if(typeof snapshot==='number'&&Number.isFinite(snapshot))target.append(svg('line',{x1:55,x2:560,y1:Y(snapshot),y2:Y(snapshot),stroke:'#64748b','stroke-dasharray':'5 4'}));
        path(values,'#4f46e5');if(prepared.length)path(prepared,'#d97706');
        const info=$('chunkInfo');info.replaceChildren();const status=detail.chunks.find(c=>c.chunk_id===data.chunk_id);
        info.append(node('p',`${statusNames[status?.status]||status?.status||'仅有预测'} · 推理 ${(status?.inference_ms??0).toFixed(1)}ms · ${data.chunk_id}`));
        info.append(node('p','观测源时间戳（states）：'+(data.observation?.timestamps_raw?.states??'未记录')));
        const exp=node('details'),summary=node('summary','查看时间戳、原始预测与发送记录');exp.append(summary,node('pre',JSON.stringify(chunk,null,2)));info.append(exp);
    }
    function renderEvents(){const body=$('eventRows');body.replaceChildren();const events=detail.events.filter(e=>e.elapsed_s>=view[0]&&e.elapsed_s<=view[1]);
        for(const event of events){const row=node('tr');let description=event.reason?(reasonNames[event.reason]||event.reason):'';
            if(event.kind==='vr_edge')description=(event.side==='left'?'左手':'右手')+(event.active?'按下':'松开');
            if(event.kind==='human_round_started')description='第'+event.round+'轮'+(event.limit_exceeded?' · 超过上限':'');
            if(event.kind==='dispatch_started')description=event.dispatch_kind==='model_chunk'?'模型动作块':'初始化 / 复位';
            if(event.kind==='result_decided')description=event.result?.success?'成功':(event.result?.reasons||[]).map(r=>reasonNames[r]||r).join('、');
            if(event.error)description+=' '+event.error;
            row.append(node('td',fmt(event.elapsed_s)),node('td',eventNames[event.kind]||event.kind),node('td',description));body.append(row);
        }
        if(!events.length){const row=node('tr'),cell=node('td','该窗口无事件');cell.colSpan=3;row.append(cell);body.append(row);}
    }
    function openEpisode(id){currentId=id;for(const key of ['dataTaskFilter','dataDateFilter','dataResultFilter','dataHumanFilter'])$(key).value='';return refresh();}
    return {refresh,filter,more:()=>refresh(true),select,round,stepRound,full,ending,zoom,pan,drawChunk,openEpisode};
})();

function viewCurrentEpisodeData(){const id=dialogContext?.kind==='episode'?dialogContext.id:serverState.episode?.id;if(!id)return;closeInterventionDialog();selectTab('data');DataBrowser.openEpisode(id);}
