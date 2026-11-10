"""Review robot-local sessions; retain read access to legacy host records."""
from pathlib import Path
from urllib.parse import urlencode
from chunk_log import Transport
from record_browser import RecordBrowser, ID

class RemoteRecordBrowser:
    def __init__(self, legacy_root, transport=None):
        self.legacy = RecordBrowser(legacy_root)
        self.transport = transport or Transport()

    def is_legacy(self, sid):
        if not isinstance(sid,str) or not ID.fullmatch(sid): raise ValueError("无效任务ID")
        return (self.legacy.root / sid).is_dir()

    def listing(self, params):
        filters={k:v[0] for k,v in params.items()}
        offset=max(0,int(filters.get("offset") or 0));limit=max(1,min(100,int(filters.get("limit") or 50)))
        old=self.legacy.listing({**params,"offset":["0"],"limit":["100"]})
        for row in old["records"]:row["legacy"]=True
        rows=list(old["records"]);errors=list(old.get("errors",[]))
        try:
            begin=0
            while begin<5000:
                page=self.transport.request("GET","/api/records?"+urlencode({**filters,"offset":begin,"limit":100}))
                rows.extend(page["records"]);errors.extend(page.get("errors",[]))
                begin+=len(page["records"])
                if begin>=page["total"] or not page["records"]:break
        except OSError as exc:
            errors.append({"episode_id":"","message":"机器人记录不可达："+str(exc)})
        rows.sort(key=lambda r:int(r.get("started_wall_ns") or 0),reverse=True)
        return {"success":True,"records":rows[offset:offset+limit],"total":len(rows),"errors":errors}

    def detail(self,sid):
        if self.is_legacy(sid):
            result=self.legacy.detail(sid);result["episode"]["legacy"]=True
            return result
        return self.transport.request("GET","/api/records/"+sid)

    def chunk(self,sid,cid):
        if self.is_legacy(sid):return self.legacy.chunk(sid,cid)
        if not ID.fullmatch(cid):raise ValueError("无效动作块ID")
        return self.transport.request("GET","/api/records/"+sid+"/chunks/"+cid)

    def write(self,sid,operation,body):
        if self.is_legacy(sid):raise ValueError("历史记录仅供查看，本次未迁移原始数据")
        if operation not in {"annotations","finalize","stop"}:raise ValueError("无效标注操作")
        if operation=="finalize":
            pending=Path(__file__).with_name("pending_spool")/sid
            if pending.exists() and any(pending.glob("*.json")):raise ValueError("还有记录待上传，暂不能归档")
        path=("/api/sessions/"+sid+"/stop") if operation=="stop" else ("/api/records/"+sid+"/"+operation)
        return self.transport.request("POST",path,body)
