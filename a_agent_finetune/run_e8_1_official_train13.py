"""Run E8_1 with the official post-trained Qwen3.5-9B checkpoint."""

from __future__ import annotations

import json
import os
import shlex
import time
from pathlib import Path

from a_agent_finetune import run_e8_base_train13 as base


RUN = base.HERE / "runs/e8_1-official-posttrain-train13-20261012-r1"
MODEL = "qwen35-9b-e8_1-official-posttrain"
REMOTE_ROOT = "/root/autodl-tmp/e8_1-official-posttrain-train13-20261012-r1"
REMOTE_MODEL = "/root/autodl-tmp/g1/models/qwen35-9b-official-posttrain"
SSH_OPTIONS = ["-o", "PreferredAuthentications=password", "-o", "PubkeyAuthentication=no"]
_original_ssh_base = base.ssh_base
_original_start_remote_service = base.start_remote_service
_original_make_inputs = base.make_inputs
base.RUN = RUN
base.MODEL = MODEL
base.REMOTE_ROOT = REMOTE_ROOT
base.REMOTE_MODEL = REMOTE_MODEL
base.SOCKET = Path("/tmp/e8_1-official-posttrain-train13-20261012.sock")


def ssh_base() -> list[str]:
    original = _original_ssh_base()
    return original[:1] + SSH_OPTIONS + original[1:]


base.ssh_base = ssh_base


def establish_control_master() -> None:
    password = os.environ.get("E8_SSH_PASSWORD")
    if not password:
        raise RuntimeError("E8_SSH_PASSWORD is required for the requested A800 instance")
    script = r'''
set timeout 40
spawn ssh -M -S $env(E8_SOCK) -o ControlPersist=12h -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PreferredAuthentications=password -o PubkeyAuthentication=no -fN -p 57563 root@connect.nma1.seetacloud.com
expect {
  -re "(?i)password:" { send "$env(E8_SSH_PASSWORD)\r"; exp_continue }
  eof {}
  timeout { exit 124 }
}
'''
    import subprocess
    last_error = None
    for attempt in range(1, 6):
        base.SOCKET.unlink(missing_ok=True)
        env = dict(os.environ, E8_SOCK=str(base.SOCKET))
        proc = subprocess.run(["expect", "-c", script], env=env, text=True)
        if proc.returncode == 0:
            try:
                base.remote("echo ssh-control-ready", timeout=30)
                return
            except Exception as error:
                last_error = error
        else:
            last_error = RuntimeError(f"control master exit={proc.returncode}")
        time.sleep(12 * attempt)
    raise RuntimeError(f"could not establish SSH control master after retries: {last_error}")


def make_inputs() -> dict[str, str]:
    state = _original_make_inputs()
    config_path = RUN / "inputs/stage_model_config.json"
    config = json.loads(config_path.read_text())
    config["default"]["model_metadata"]["name"] = (
        "Qwen3.5-9B official post-trained model with E8 repeat guard"
    )
    base.write_json(config_path, config)
    state["experiment"] = (
        "E8_1 official post-trained Qwen3.5-9B with explicit tool-repeat guard"
    )
    state["model_source"] = {
        "model_id": "Qwen/Qwen3.5-9B",
        "model_type": "official post-trained",
        "local_model_path": REMOTE_MODEL,
        "base_comparison": "Qwen/Qwen3.5-9B-Base in E8",
    }
    base.write_json(RUN / "batch.json", state)
    return state


def start_remote_service() -> None:
    # Download once into a stable local directory; reuse it on reruns.
    check = (
        f"test -f {shlex.quote(REMOTE_MODEL + '/config.json')} "
        f"&& echo model-present || echo model-missing"
    )
    if "model-present" not in base.remote(check, timeout=60):
        download = f'''set -euo pipefail
MODEL={shlex.quote(REMOTE_MODEL)}
rm -rf "$MODEL.incomplete"
mkdir -p "$MODEL.incomplete"
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_TELEMETRY=1 \
  /root/autodl-tmp/g1/venv/bin/python -c \
  'from huggingface_hub import snapshot_download; snapshot_download(repo_id="Qwen/Qwen3.5-9B", local_dir="{REMOTE_MODEL}.incomplete", local_dir_use_symlinks=False)' 
mv "$MODEL.incomplete" "$MODEL"
'''
        base.remote(download, timeout=3600)
    _original_start_remote_service()


if __name__ == "__main__":
    # The previous interrupted E7 probe is not part of this experiment.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    base.establish_control_master = establish_control_master
    base.make_inputs = make_inputs
    base.start_remote_service = start_remote_service
    base.main()
