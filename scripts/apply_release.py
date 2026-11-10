"""Deploy validated sources to an idle host or the robot's two auxiliary services.

Does not start models, configure remote_env, initialize joints, or send actions.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import py_compile
import shutil
import signal
import socket
import subprocess
import tarfile
import tempfile
import time
from urllib.request import build_opener, ProxyHandler

def read_json(url):
    with build_opener(ProxyHandler({})).open(url,timeout=3) as response:return json.load(response)
def listening(port):
    with socket.socket() as sock:return sock.connect_ex(("127.0.0.1",port))==0
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def processes(target):
    found=[]
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():continue
        try:
            argv=(path/"cmdline").read_bytes().replace(b"\0",b" ").decode(errors="replace")
            if "intergrated_test_server.py" in argv and (path/"cwd").resolve()==target:
                found.append(int(path.name))
        except OSError:pass
    return found
def start(argv,target,log,env=None):
    with (target/log).open("ab") as stream:
        proc=subprocess.Popen(argv,cwd=target,env=env,stdin=subprocess.DEVNULL,
             stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    return proc.pid

def apply(role,archive):
    target=(Path("/home/lindenbot/qianzhang43/agibot_policy_ctl/g2_policy_ctl/intergrated_test")
            if role=="host" else Path("/data/yfmo/takeover_system"))
    if role=="host":
        state=read_json("http://127.0.0.1:8084/api/config")
        if state.get("locked") or (state.get("episode") or {}).get("phase") in {"initializing","running","paused","resetting","awaiting_result"}:
            raise RuntimeError("Web task is active; deployment refused")
        python=Path("/home/lindenbot/miniconda3/envs/pi_05/bin/python")
        if not python.exists():raise RuntimeError("Expected host Python environment missing")
        if not (target/"session_receiver.token").exists():raise RuntimeError("New receiver token missing")
    else:
        if listening(8086) or listening(8087):raise RuntimeError("Auxiliary port already used; inspect before replacing")
        python=Path("/usr/bin/python3")
        if not (target/"session_receiver.token").exists():raise RuntimeError("Receiver token missing")
    with tarfile.open(archive) as tar:
        manifest=json.load(tar.extractfile("release_manifest.json"))
        names=[n for n in manifest if n.startswith(role+"/")]
        if sum(manifest[n]["bytes"] for n in names)>50_000_000:raise RuntimeError("Oversized release")
        payloads={}
        for name in names:
            if Path(name).name!=name.split("/",1)[1]:raise RuntimeError("Unsafe archive path")
            data=tar.extractfile(name).read()
            if len(data)!=manifest[name]["bytes"] or hashlib.sha256(data).hexdigest()!=manifest[name]["sha256"]:raise RuntimeError("Release digest mismatch")
            existing=target/Path(name).name
            if existing.is_symlink():raise RuntimeError("Unexpected symlink destination")
            previous=manifest[name]["previous_sha256"]
            if existing.exists() and sha(existing)!=manifest[name]["sha256"]:
                if role=="robot" or previous is None or sha(existing)!=previous:
                    raise RuntimeError("User source changed: "+str(existing))
            payloads[existing]=data
    target.mkdir(parents=True,exist_ok=True)
    backup=target.parent/(target.name+"_backup_"+str(time.time_ns()))
    backup.mkdir()
    existed={}
    for path in payloads:
        existed[path]=path.exists()
        if path.exists():shutil.copy2(path,backup/path.name)
    if role=="host" and (target/"web_settings.json").exists():shutil.copy2(target/"web_settings.json",backup/"web_settings.json")
    for path,data in payloads.items():
        if path.suffix==".py":compile(data,str(path),"exec")
    if role=="host":
        state=read_json("http://127.0.0.1:8084/api/config")
        if state.get("locked"):raise RuntimeError("Task started during deployment preflight")
        settings=dict(state["config"])
        old=processes(target)
        if not old:raise RuntimeError("Cannot identify current web process")
        for pid in old:os.kill(pid,signal.SIGTERM)
        deadline=time.monotonic()+4
        while listening(8084) and time.monotonic()<deadline:time.sleep(.1)
    for path,data in payloads.items():
        temporary=path.with_suffix(path.suffix+".release")
        temporary.write_bytes(data);os.replace(temporary,path)
    for path in payloads:
        if path.suffix==".py":py_compile.compile(str(path),doraise=True)
    if role=="robot":
        env={**os.environ,"LD_LIBRARY_PATH":"/home/agi/app/lib:/home/agi/app/gdk/lib:"+os.environ.get("LD_LIBRARY_PATH","")}
        monitor=start([str(python),"-u",str(target/"vr_monitor.py"),"--host","0.0.0.0"],target,"monitor.log",env)
        receiver=start([str(python),"-u",str(target/"collection_server.py"),"--host","0.0.0.0","--token-file",str(target/"session_receiver.token")],target,"collection.log",env)
        time.sleep(2)
        if not listening(8086) or not listening(8087):raise RuntimeError("Auxiliary startup failed; inspect logs")
        print(json.dumps({"role":role,"backup":str(backup),"monitor_pid":monitor,"collection_pid":receiver,"motion_sent":False}))
    else:
        settings["human_takeover_enabled"]=True
        (target/"web_settings.json").write_text(json.dumps(settings,ensure_ascii=False,indent=2))
        env={**os.environ,"NO_PROXY":"192.168.50.113,192.168.50.61,127.0.0.1,localhost",
             "no_proxy":"192.168.50.113,192.168.50.61,127.0.0.1,localhost"}
        pid=start([str(python),"-u",str(target/"intergrated_test_server.py")],target,"takeover_v2.log",env)
        time.sleep(2)
        if not listening(8084):
            # Restore only the files this release touched, then restore the old preferences.
            for path,was_present in existed.items():
                if was_present:shutil.copy2(backup/path.name,path)
                else:path.unlink(missing_ok=True)
            if (backup/"web_settings.json").exists():shutil.copy2(backup/"web_settings.json",target/"web_settings.json")
            rollback_pid=start([str(python),"-u",str(target/"intergrated_test_server.py")],target,"rollback.log",env)
            raise RuntimeError("New web startup failed; previous sources restored, pid="+str(rollback_pid))
        print(json.dumps({"role":role,"backup":str(backup),"web_pid":pid,"motion_sent":False}))
if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role",choices=["host","robot"]);parser.add_argument("archive")
    args=parser.parse_args();apply(args.role,args.archive)
