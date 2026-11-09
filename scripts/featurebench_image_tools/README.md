# FeatureBench cache-only image prototype

These standalone tools create and validate a new image. They do not update an
evaluation manifest, replace an official image tag, or alter the running batch.

`strip_caches.py` reads a single-image Docker-save archive and removes only:

- `/root/.cache/pip/`
- `/opt/miniconda3/pkgs/cache/`
- downloaded `.conda` and `.tar.bz2` archives directly under Conda's `pkgs/`

Extracted Conda packages, both environments, CUDA, system tools, project sources,
and source backups remain. A runtime link into a removed cache aborts conversion.
Every retained file's content and tar metadata is checked. Unchanged layers are
copied exactly; rewritten layers retain their boundaries, preserving the ability
to share identical base layers across project images.

Unlike deleting files in an additional Dockerfile layer, rewriting the original
layers removes their storage cost. The resulting image has a distinct ID and tag.
Deleting caches also changes the resources available to later offline package
installation; passing the validation below does not prove all solver behavior is
unchanged.

Example (use dedicated paths with sufficient free disk space):

```sh
python3 scripts/featurebench_image_tools/strip_caches.py \
  --source /path/to/single-image.docker.tar \
  --workdir /path/to/work \
  --tag external-eval-prototypes/featurebench-cache-only:trial \
  --docker-host unix:///path/to/isolated-docker.sock \
  --target-storage /path/to/docker-filesystem \
  --archive-output /path/to/candidate.docker.tar
```

The source must contain ordinary uncompressed Docker-save layer tar files. The
output uses compressed inner layers, verified with Docker 26.1.1. Budget for both
Docker's temporary load/export staging and uncompressed image storage; the tool
does not make a large image fit into insufficient disk space. Its space guard
stops only its own loader. A failed load can leave partial unreferenced Docker
layers, which should be inspected individually rather than globally pruned.

`validate.py` is the server1 Packaging prototype's run-specific validation driver:
it checks image configuration, inventories both Python environments, and compares
the official gold and negative-control evaluations on the original and candidate.
After successful loading it removes its own temporary source archive. After all
checks pass it removes only the candidate tag from the evaluation daemon, retaining
the compact candidate archive and audit reports. It makes zero model API calls.

Tests:

```sh
python3 -m unittest discover -s tests -p test_featurebench_image_tools.py -v
```
