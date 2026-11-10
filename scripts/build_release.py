"""Build an explicit small deployment artifact; never package credentials or raw data."""
import hashlib
import json
from pathlib import Path
import tarfile
ROOT=Path(__file__).resolve().parents[1]
HOST_NAMES=("chunk_log.py","episode_control.py","data_browser.css","data_browser.js",
    "intergrated_test.html","intergrated_test_server.py","takeover_runtime.py",
    "washer_adapter.py","washer_tasks.py","vr_proxy.py","network_config.py",
    "network.json","remote_records.py","web_config.py")
ROBOT_NAMES=("collection_server.py","storage.py","native_collector.py","record_view.py",
    "vr_monitor.py","vr_core.py")
def build():
    baseline=json.loads((ROOT/"deploy/source_baseline.json").read_text())
    manifest={}
    for role,names in [("host",HOST_NAMES),("robot",ROBOT_NAMES)]:
        for name in names:
            path=ROOT/role/name
            manifest[role+"/"+name]={"bytes":path.stat().st_size,
                "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                "previous_sha256":baseline.get(role+"/"+name,{}).get("sha256")}
    total=sum(v["bytes"] for v in manifest.values())
    if total>50_000_000:raise RuntimeError("Release needs explicit transfer-size approval")
    (ROOT/"deploy/release_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    output=ROOT/"runtime/release.tar"
    with tarfile.open(output,"w") as archive:
        for name in manifest:archive.add(ROOT/name,arcname=name)
        archive.add(ROOT/"deploy/release_manifest.json",arcname="release_manifest.json")
    print(json.dumps({"source_bytes":total,"archive_bytes":output.stat().st_size,
        "archive_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),"path":str(output)}))
if __name__=="__main__":build()
