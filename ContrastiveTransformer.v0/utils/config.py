import json

def read_config(json_file):
    with open(json_file, 'r') as f:
        config = json.load(f)
    return config

