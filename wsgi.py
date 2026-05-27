from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


APP_DIR = Path(__file__).resolve().parent
APP_PATH = APP_DIR / "dashboard_app.py"

spec = spec_from_file_location("dashboard_app_heroku", APP_PATH)
if spec is None or spec.loader is None:
	raise RuntimeError(f"No se pudo cargar la app desde {APP_PATH}")

module = module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

app = module.APP