import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_tmp = Path(tempfile.mkdtemp(prefix="ragly-test-"))
engines = json.loads((ROOT / "engines.json").read_text())
engines["normal"]["base_url"] = "http://127.0.0.1:1/v1"   # nothing listens here
engines["normal"]["manage_process"] = False
(_tmp / "engines.json").write_text(json.dumps(engines))
os.environ["RAGLY_DATA"] = str(_tmp / "data")
os.environ["RAGLY_ENGINES"] = str(_tmp / "engines.json")
