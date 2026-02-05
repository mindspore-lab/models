import json
import os
import pandas as pd
import logging

logger = logging.getLogger('eval_logger')

def read_jsonl(file_path):
    """Reads JSONL, handles directories recursively."""
    data_list = []
    if os.path.isdir(file_path):
        for root, _, files in os.walk(file_path):
            for file_name in files:
                if file_name.endswith('.jsonl'):
                    data_list.extend(read_jsonl_single(os.path.join(root, file_name)))
    else:
        data_list = read_jsonl_single(file_path)
    return data_list

def read_jsonl_single(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.error(f"JSON Parse Error in {file_path}")
    except Exception as e:
        logger.error(f"Read Error {file_path}: {e}")
    return data

def write_jsonl(file_path, data_list):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def write_excel(file_path, data_dict):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df = pd.DataFrame(data_dict)
    df.to_excel(file_path, index=False, engine='openpyxl')