"""Send credentials through SSH stdin and launch one detached, idempotent smoke batch."""
from __future__ import annotations

import argparse
import getpass
import json
import shlex
import subprocess

REMOTE_LAUNCH = r'''
import hashlib,json,os,subprocess,sys,urllib.request
from datetime import datetime,timezone
from pathlib import Path
request=json.load(sys.stdin)
secret=request.pop("api_key")
root=Path(request["root"])
project=root/"project"
run=root/"runs"/request["run_name"]
record=run/"launch.json"
if record.exists():
 print(json.dumps({"status":"already_launched",**json.loads(record.read_text())}));sys.exit(0)
model_config=json.loads((project/"configs/v6.1-deepseek-v4.1-flash.json").read_text())["default"]
auth_request=urllib.request.Request(model_config["base_url"].rstrip("/")+"/models",
                                    headers={"Authorization":"Bearer "+secret})
with urllib.request.urlopen(auth_request,timeout=30) as response:
 models=json.load(response)
if model_config["model"] not in {item["id"] for item in models.get("data",[])}:
 raise ValueError("Configured model is not listed by the authenticated provider endpoint")
run.mkdir(parents=True,exist_ok=False)
for name in ("manifests","grader","venv312","project"):
 (run/name).symlink_to(root/name,target_is_directory=True)
workers=min(2,len(request["task_ids"]))
command=[str(root/"venv312/bin/python"),str(project/"scripts/swe_verified_v62/run.py"),
         "generate","--root",str(run),"--project",str(project),"--workers",str(workers)]
for task in request["task_ids"]:command += ["--task-id",task]
if request.get("reuse_a_from"):
 plan={"task_id":request["task_ids"][0],"start_at":"Solver","source_run":request["reuse_a_from"]}
 (run/"fork.json").write_text(json.dumps(plan,indent=2))
 command += ["--fork-plan",str(run/"fork.json")]
environment=os.environ.copy()
environment.update(DEEPSEEK_API_KEY=secret,PYTHONUNBUFFERED="1",DO_NOT_TRACK="1",
                   LITELLM_LOCAL_MODEL_COST_MAP="True")
with (run/"launcher.log").open("ab",buffering=0) as log:
 process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                          env=environment,cwd=project,start_new_session=True)
meta={"status":"launched","pid":process.pid,"root":str(run),"task_ids":request["task_ids"],
      "workers":workers,"framework":"mini-swe-agent==2.4.6","workflow":"v6.2",
      "reuse_a_from":request.get("reuse_a_from"),
      "model":"deepseek-flash","started_at":datetime.now(timezone.utc).isoformat(),
      "command":command,"credentials_persisted":False,
      "authentication_preflight":"GET /models succeeded; configured model listed",
      "source_sha256":{str(p.relative_to(project)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (project/"external_agents_eval/agents/mini_v62.py",
                                 project/"external_agents_eval/agents/mini_runtime_assets.py",
                                 project/"external_agents_eval/agents/pier/pi_v6_solve_review_repair.py",
                                 project/"external_agents_eval/workflows/solve_review_repair/v62_controller.py",
                                 project/"external_agents_eval/workflows/solve_review_repair/ft/prompt.txt")}}
record.write_text(json.dumps(meta,indent=2))
print(json.dumps(meta))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-path", required=True)
    parser.add_argument("--run-name", default="smoke2-20261026-r1")
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--reuse-a-from")
    args = parser.parse_args()
    tasks = args.task_id or ["astropy__astropy-13453", "django__django-10999"]
    if args.reuse_a_from and len(tasks) != 1:
        parser.error("--reuse-a-from requires exactly one --task-id")
    secret = getpass.getpass("DeepSeek API key (not saved): ")
    request = {"root": "/root/v62-swe-verified100-20261026", "run_name": args.run_name,
               "task_ids": tasks, "reuse_a_from": args.reuse_a_from, "api_key": secret}
    command = ["ssh", "-S", args.control_path, "-p", "25944", "root@223.109.239.11",
               "python3 -c " + shlex.quote(REMOTE_LAUNCH)]
    result = subprocess.run(command, input=json.dumps(request), text=True, capture_output=True, check=False)
    print(result.stdout.replace(secret, "[REDACTED]"))
    if result.stderr:
        print(result.stderr.replace(secret, "[REDACTED]"))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
