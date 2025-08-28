
import yaml, pathlib

def load_yaml(path):
    path = pathlib.Path(path)
    with path.open("r") as f:
        return yaml.safe_load(f)

def ensure_dir(p):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)
