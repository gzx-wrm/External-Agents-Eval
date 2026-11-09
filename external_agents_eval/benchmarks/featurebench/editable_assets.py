"""Give role worktrees the compiled outputs of a meson-python editable install.

Standalone (stdlib only); it runs inside task containers with the image's Python.

A meson-python editable install registers a meta-path finder that maps the package
to the main checkout and its build directory, whatever the working directory or
PYTHONPATH. Role worktrees therefore cannot import their own copy. ``scan`` finds such
finders whose build directory lives in the repository, brings that build up to date
with the finder's own build command (exactly what the finder does on first import),
and records the build outputs from meson's install plan. ``seed`` copies those outputs
into a worktree at their installed package paths and keeps them out of Git diffs.
Role commands then set the finder's own skip marker so imports use the worktree.
"""
import argparse
import ast
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sysconfig

STATE = "/external_eval/editable-assets/scan.json"
EXCLUDE_HEADER = "# featurebench_v6.4: meson-python editable build outputs seeded into worktrees"


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def finders():
    found = []
    roots = sorted({sysconfig.get_paths()["purelib"], sysconfig.get_paths()["platlib"]})
    for root in roots:
        for path in sorted(glob.glob(os.path.join(root, "_*_editable_loader.py"))):
            with open(path, encoding="utf-8") as stream:
                text = stream.read()
            if "MesonpyMetaFinder" not in text:
                continue
            marker, calls = None, []
            for node in ast.parse(text).body:
                if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "MARKER" for t in node.targets):
                    marker = ast.literal_eval(node.value)
                if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                        and getattr(node.value.func, "id", None) == "install"):
                    args = [ast.literal_eval(argument) for argument in node.value.args]
                    calls.append({"names": sorted(args[0]), "build_path": args[1], "build_cmd": list(args[2])})
            for call in calls:
                found.append(dict(loader=path, marker=marker, **call))
    return found


def walk(root):
    for directory, _, files in os.walk(root):
        for name in files:
            yield os.path.join(directory, name)


def outputs(build_path, plan):
    """Install-plan entries produced in the build directory, as (source, package-relative path)."""
    prefix = build_path.rstrip("/") + "/"
    targets = []
    for key, data in plan.items():
        for source, target in data.items():
            parts = target["destination"].split("/")
            if parts[0] not in ("{py_platlib}", "{py_purelib}") or not source.startswith(prefix):
                continue
            relative = "/".join(parts[1:])
            if key == "install_subdirs" and os.path.isdir(source):
                for path in walk(source):
                    targets.append((path, relative + "/" + os.path.relpath(path, source)))
            elif os.path.isfile(source):
                targets.append((source, relative))
    return sorted(set(targets), key=lambda item: item[1])


def scan(repository, rebuild):
    prefix = repository.rstrip("/") + "/"
    builds = []
    for finder in finders():
        build_path = finder["build_path"]
        plan_path = os.path.join(build_path, "meson-info", "intro-install_plan.json")
        if not build_path.startswith(prefix) or not os.path.isfile(plan_path) or not finder["marker"]:
            continue
        if rebuild:
            env = dict(os.environ)
            env[finder["marker"]] = os.pathsep.join(filter(None, [env.get(finder["marker"]), build_path]))
            subprocess.run(finder["build_cmd"], cwd=build_path, env=env, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        with open(plan_path, encoding="utf-8") as stream:
            plan = json.load(stream)
        targets = [{"source": source, "path": relative, "sha256": digest(source)}
                   for source, relative in outputs(build_path, plan)]
        builds.append(dict(finder, targets=targets))
    state = {"repository": repository, "builds": builds}
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    with open(STATE, "w", encoding="utf-8") as stream:
        json.dump(state, stream)
    return {"repository": repository, "builds": [
        {"names": b["names"], "build_path": b["build_path"], "marker": b["marker"],
         "targets": len(b["targets"])} for b in builds]}


def git(worktree, *args, data=None):
    return subprocess.run(["git", "-C", worktree, *args], input=data, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False)


def seed(worktree):
    with open(STATE, encoding="utf-8") as stream:
        state = json.load(stream)
    report = {"worktree": worktree, "files": 0, "bytes": 0, "excluded_locally": []}
    for build in state["builds"]:
        top = build["names"][0].split(".")[0] if build["names"] else None
        base = worktree
        if top and not os.path.isdir(os.path.join(worktree, top)) and os.path.isdir(os.path.join(worktree, "src", top)):
            base = os.path.join(worktree, "src")
        relatives = []
        for target in build["targets"]:
            if digest(target["source"]) != target["sha256"]:
                raise ValueError("Build output changed after scan: " + target["source"])
            destination = os.path.join(base, target["path"])
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copy2(target["source"], destination)
            relatives.append(os.path.relpath(destination, worktree))
            report["files"] += 1
            report["bytes"] += os.path.getsize(destination)
        # Generated outputs that the project's .gitignore does not cover are kept out of
        # every diff through the repository-local exclude file (never committed).
        checked = git(worktree, "check-ignore", "--stdin", data="\n".join(relatives).encode())
        ignored = set(checked.stdout.decode().splitlines())
        missing = [path for path in relatives if path not in ignored]
        if missing:
            common = git(worktree, "rev-parse", "--git-common-dir").stdout.decode().strip()
            exclude = os.path.join(worktree, common, "info", "exclude") if not os.path.isabs(common) \
                else os.path.join(common, "info", "exclude")
            os.makedirs(os.path.dirname(exclude), exist_ok=True)
            existing = open(exclude, encoding="utf-8").read() if os.path.exists(exclude) else ""
            lines = [line for line in ("/" + path for path in missing) if line not in existing.splitlines()]
            if lines:
                with open(exclude, "a", encoding="utf-8") as stream:
                    if EXCLUDE_HEADER not in existing:
                        stream.write(("\n" if existing and not existing.endswith("\n") else "") + EXCLUDE_HEADER + "\n")
                    stream.write("\n".join(lines) + "\n")
            report["excluded_locally"].extend(missing)
        still = git(worktree, "status", "--porcelain", "--untracked-files=all", "--", *relatives)
        if still.stdout.strip():
            raise ValueError("Seeded build outputs are visible to Git: " + still.stdout.decode()[:2000])
    return report


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    scan_parser = commands.add_parser("scan")
    scan_parser.add_argument("--repository", required=True)
    scan_parser.add_argument("--rebuild", action="store_true")
    seed_parser = commands.add_parser("seed")
    seed_parser.add_argument("--worktree", required=True)
    args = parser.parse_args()
    if args.command == "scan":
        result = scan(args.repository, args.rebuild)
    else:
        result = seed(args.worktree)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
