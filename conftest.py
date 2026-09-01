import sys
from pathlib import Path

# The package `dq_systems` *is* this repository (this file's directory), so
# for `import dq_systems` to resolve, the repo's parent directory must be on
# sys.path. A root-level conftest.py is always imported by pytest before
# test collection, so doing it here means tests work regardless of the
# directory pytest is invoked from, without needing the package installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
