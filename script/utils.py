import json, os

def load_config(filename="config.json"):
    # 当前文件(script/utils.py)的上一级目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, filename)
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)
