# import json
import yaml


# def read_config(json_file):
#     with open(json_file, 'r') as f:
#         config = json.load(f)
#     return config

def read_config(yml_file):
    with open(yml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config