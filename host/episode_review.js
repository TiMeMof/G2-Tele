'use strict';
const EpisodeReview=(()=>{
    const $=id=>document.getElementById(id);
    let draft=null,opening=false,saving=false;
    const label=s=>Number.isFinite(s)?s.toFixed(2)+'s':'未知';
    function node(tag,text){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;return n;}
    async function api(url,body){
        const response=await fetch(url,{method:body?'POST':'GET',headers:body?{'Content-Type':'application/json'}:{},
            body:body?JSON.stringify(body):undefined,signal:AbortSignal.timeout(60000)});
        const value=await response.json();if(!response.ok||!value.success)throw new Error(value.message||'操作失败');return value;
    }
    function valid(item){return !!((item.reasons.length||item.note.trim())&&(!item.reasons.includes('other')||item.note.trim()));}
    function ensurePanel(){
        if($('inlineReview'))return;
        const panel=node('section');panel.id='inlineReview';panel.hidden=true;
        panel.style.cssText='border-top:1px solid #e2e8f0;margin-top:18px;padding-top:16px';
        panel.append(node('h3','失败任务 · 接管段落标注'));
        const status=node('p');status.id='reviewStatus';status.setAttribute('role','status');panel.append(status);
        const retry=node('button','重试结束录制');retry.id='reviewRetryStop';retry.type='button';retry.className='btn-secondary';retry.onclick=()=>open();panel.append(retry);
        const rounds=node('div');rounds.id='reviewRounds';rounds.style.cssText='display:flex;gap:8px;flex-wrap:wrap;margin:14px 0';panel.append(rounds);
        const editor=node('section');editor.id='reviewEditor';panel.append(editor);
        const overall=node('label','整条失败补充说明（没有接管段时必填）');
        const note=node('textarea');note.id='reviewOverallNote';note.maxLength=2000;note.rows=2;note.style.cssText='display:block;width:100%;margin:8px 0';
        note.oninput=()=>{if(draft)draft.note=note.value;};overall.append(note);panel.append(overall);
        const message=node('p');message.id='reviewMessage';message.setAttribute('role','status');panel.append(message);
        const save=node('button','保存本条数据');save.id='reviewSave';save.type='button';save.className='btn-primary';save.onclick=()=>submit();panel.append(save);
        panel.append(node('p','填写内容为当前页面草稿；点击保存后才提交段落原因、失败结果并归档。'));
        $('episodeRecording').before(panel);
    }
    function drawRounds(){
        const list=$('reviewRounds');list.replaceChildren();
        if(!draft.rounds.length)list.append(node('p','本条没有人工接管段，请填写整条失败说明。'));
        draft.rounds.forEach((item,index)=>{
            const button=node('button',`第 ${item.round} 段 · ${label(item.start_s)}–${label(item.end_s)} · ${valid(item)?'已填写':'待填写'}`);
            button.type='button';button.dataset.round=String(item.round);button.setAttribute('aria-pressed',String(index===draft.selected));
            button.style.cssText=`padding:9px 12px;border-radius:9px;border:2px solid ${index===draft.selected?'#4f46e5':'#fecaca'};background:${valid(item)?'#ecfdf5':'#fff1f2'};color:#881337;cursor:pointer`;
            button.disabled=saving||!!draft.submitted;button.onclick=()=>{draft.selected=index;drawRounds();drawEditor();};list.append(button);
        });
    }
    function drawEditor(){
        const editor=$('reviewEditor');editor.replaceChildren();const item=draft.rounds[draft.selected];if(!item)return;
        editor.append(node('h4','第 '+item.round+' 次接管：为什么接管、纠正了什么？'));
        const choices=node('div');choices.style.cssText='display:flex;flex-wrap:wrap;gap:12px;margin:12px 0';
        for(const [id,name] of Object.entries(draft.options)){
            const wrapper=node('label');const input=node('input');input.type='checkbox';input.value=id;input.checked=item.reasons.includes(id);
            input.disabled=saving||!!draft.submitted;input.onchange=()=>{item.reasons=[...choices.querySelectorAll('input:checked')].map(n=>n.value);drawRounds();$('reviewMessage').textContent='草稿待保存。';};
            wrapper.append(input,document.createTextNode(' '+name));choices.append(wrapper);
        }
        editor.append(choices);const note=node('textarea');note.id='reviewRoundNote';note.rows=3;note.maxLength=2000;note.value=item.note;
        note.placeholder='也可以只写具体原因；选择“其他”时需填写说明。';note.style.cssText='display:block;width:100%;margin:8px 0';
        note.disabled=saving||!!draft.submitted;note.oninput=()=>{item.note=note.value;drawRounds();$('reviewMessage').textContent='草稿待保存。';};editor.append(note);
    }
    async function open(){
        if(dialogContext?.kind==='legacy'||!serverState.capabilities?.inline_review_v1){$('failureForm').hidden=false;return;}
        const ep=serverState.episode;if(!ep||opening||saving)return;
        opening=true;ensurePanel();$('failureForm').hidden=true;
        if(draft?.id!==ep.id)draft={id:ep.id,rounds:[],selected:0,note:'',options:{},loaded:false,submitted:null};
        $('inlineReview').hidden=false;$('reviewMessage').textContent='正在结束本条录制并读取接管段落…';render(ep);
        try{
            await api('/api/episode/command',{command:'review',episodeId:ep.id,revision:ep.revision,requestId:Date.now()+'-'+Math.random()});
            const data=await api('/api/records/'+ep.id);const d=data.episode;
            const old=new Map(draft.rounds.map(r=>[r.round,r]));
            draft.rounds=d.rounds.map(r=>old.get(r.round)||{...r,reasons:r.annotation?.reasons||[],note:r.annotation?.note||''});
            if(ep.review_submission){
                const submitted=ep.review_submission;const saved=new Map(submitted.round_annotations.map(r=>[r.round,r]));
                draft.rounds=draft.rounds.map(r=>({...r,...saved.get(r.round)}));draft.note=submitted.note;
                draft.submitted={episodeId:ep.id,...submitted};
            }
            draft.options=d.reason_options||serverState.takeover_readiness?.failure_reasons||{};draft.loaded=true;
            $('reviewOverallNote').value=draft.note;drawRounds();drawEditor();
            $('reviewMessage').textContent='逐段点击红色段落填写原因，全部完成后点击保存。';
        }catch(e){$('reviewMessage').textContent='尚未完成：'+e.message;}
        finally{opening=false;await loadConfig();render(serverState.episode);}
    }
    function render(ep){
        ensurePanel();const active=!!ep&&(ep.review_pending||draft?.id===ep.id&&!ep.review_saved);
        if(ep?.review_saved&&draft?.id===ep.id){draft=null;closeInterventionDialog();$('episodeMessage').textContent='本条数据已保存并归档。';}
        $('inlineReview').hidden=!active;if(!active)return false;
        if(!draft||draft.id!==ep.id){if(!opening)setTimeout(()=>open(),0);return true;}
        $('failureForm').hidden=true;
        $('episodeHelp').textContent='本条任务已结束。逐段填写接管原因，点击保存后提交失败结果与全部标注。';
        const stopped=ep.recording?.collection_stopped;
        $('reviewStatus').textContent=serverState.recording_error?'录制收尾异常：'+serverState.recording_error:stopped?(ep.review_ready?'录制已结束 · 标注待保存':'录制已结束 · 正在等待记录收尾…'):'正在结束录制…';
        $('reviewRetryStop').hidden=!!stopped&&!serverState.recording_error;$('reviewRetryStop').disabled=opening||saving;
        $('reviewSave').disabled=opening||saving||!draft.loaded||!ep.review_ready;
        $('reviewSave').textContent=saving?'正在保存…':draft.submitted?'重试保存':'保存本条数据';
        $('reviewOverallNote').disabled=saving||!!draft.submitted;
        return true;
    }
    async function submit(){
        if(!draft||saving)return;
        const ep=serverState.episode;if(!ep||ep.id!==draft.id)return;
        const missing=draft.rounds.filter(r=>!valid(r));
        if(missing.length){draft.selected=draft.rounds.indexOf(missing[0]);drawRounds();drawEditor();$('reviewMessage').textContent='请补齐第 '+missing.map(r=>r.round).join('、')+' 段原因。';return;}
        if(!draft.rounds.length&&!draft.note.trim()){$('reviewMessage').textContent='请填写整条失败说明。';return;}
        draft.submitted ||= {episodeId:draft.id,round_annotations:draft.rounds.map(r=>({round:r.round,reasons:r.reasons,note:r.note})),reasons:[],note:draft.note};
        saving=true;render(ep);drawRounds();drawEditor();
        try{
            await api('/api/episode/review',draft.submitted);
            $('reviewMessage').textContent='失败结果及所有接管段已保存并归档。';
            showNotification('本条数据已保存并归档','success');closeInterventionDialog();
        }catch(e){$('reviewMessage').textContent='保存未完成，草稿保留，请点击重试保存：'+e.message;}
        finally{saving=false;await loadConfig();render(serverState.episode);}
    }
    return {open,render};
})();
