"""Install one pinned public IERS data file without changing package code."""
from __future__ import annotations

import hashlib
import io
import json
import re
import shlex
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from common import save


def verified_data(configuration):
    settings = json.loads(configuration.read_text())
    path = (configuration.parent / settings["file"]).resolve()
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != settings["sha256"]:
        raise ValueError("Pinned IERS data SHA256 differs")
    match = re.search(r"^#\s*File expires on (.+)$", data.decode(), re.MULTILINE)
    expires = datetime.strptime(match[1].strip(), "%d %B %Y").replace(tzinfo=timezone.utc)
    if expires.isoformat() != settings["expires"] or expires <= datetime.now(timezone.utc):
        raise ValueError("Pinned official IERS data has expired or its metadata differs")
    return settings, data


def install_leap_seconds(container, configuration, receipt):
    configuration = Path(configuration)
    if not configuration.exists():
        return None
    settings, data = verified_data(configuration)
    prefix = "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && "

    def execute(program):
        code, output = container.exec_run(["bash", "-lc", prefix + "python -c " + shlex.quote(program)])
        if code:
            raise RuntimeError("IERS data installation failed: " + output.decode(errors="replace"))
        return json.loads(output.decode().splitlines()[-1])

    original = execute("import astropy_iers_data as d,hashlib,json; from pathlib import Path; "
                       "p=Path(d.IERS_LEAP_SECOND_FILE); "
                       "print(json.dumps({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))")
    target = Path(original["path"])
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        info = tarfile.TarInfo(target.name)
        info.size, info.mode = len(data), 0o644
        tar.addfile(info, io.BytesIO(data))
    if not container.put_archive(str(target.parent), archive.getvalue()):
        raise RuntimeError("Docker rejected the pinned IERS data file")
    installed = execute("import hashlib,json; from pathlib import Path; "
                        f"print(json.dumps(hashlib.sha256(Path({str(target)!r}).read_bytes()).hexdigest()))")
    if installed != settings["sha256"]:
        raise ValueError("Container IERS data SHA256 differs after installation")
    record = {"container": container.name, "source": settings["source"], "sha256": installed,
              "expires": settings["expires"], "destination": str(target),
              "original_sha256": original["sha256"], "package_version_changed": False,
              "source_code_changed": False, "tests_changed": False}
    save(receipt, record)
    return record
