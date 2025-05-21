import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm

def unify_label(label: str) -> int:
    label = str(label).lower()
    if label in ["positive", "pos", "2"]:
        return 2
    elif label in ["neutral", "neu", "1"]:
        return 1
    elif label in ["negative", "neg", "0"]:
        return 0
    else:
        raise ValueError(f"Unknown label: {label}")

def load_semeval_dir(folder_path: str, output_folder: str):
    output_folder = Path(output_folder) / "laptops"
    folder = Path(folder_path)
    output = Path(output_folder)
    output.mkdir(parents=True, exist_ok=True)
    for file in tqdm(list(folder.glob("*.csv")), desc="Processing SemEval"):
        if file.name not in {
            "Laptop_Train_v2.csv",
            "Laptops_Test_Data_PhaseA.csv",
            "Laptops_Test_Data_PhaseB.csv"
        }:
            continue

        data = []
        with open(file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            if headers is None:
                continue

            is_labeled = "Aspect Term" in headers and "polarity" in headers

            for row in reader:
                text = row.get("Sentence")
                if is_labeled:
                    aspect = row.get("Aspect Term")
                    polarity = row.get("polarity")
                    if text and aspect and polarity:
                        try:
                            label = unify_label(polarity)
                            data.append({"text": text, "aspect": aspect, "label": label})
                        except ValueError:
                            continue
                else:
                    if text:
                        data.append({
                            "text": text,
                            "aspect": None,
                            "label": None,
                            "inference_only": True
                        })
        save_to_json(data, output / file.with_suffix(".json").name)

def load_ecare_dir(folder_path: str, output_folder: str):
    folder = Path(folder_path)
    output = Path(output_folder)
    output.mkdir(parents=True, exist_ok=True)

    for file in tqdm(list(folder.glob("*.jsonl")), desc="Processing e-CARE"):
        data = []
        with open(file, encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    premise = entry.get("premise")
                    h1 = entry.get("hypothesis1")
                    h2 = entry.get("hypothesis2")
                    label = entry.get("label")
                    ask_for = entry.get("ask-for", "")

                    if premise and h1 and h2 and label is not None:
                        # 正确的 hypothesis → label=1，错误的 → label=0
                        data.append({
                            "text": f"{premise} [SEP] {h1}",
                            "aspect": ask_for,
                            "label": 1 if label == 0 else 0
                        })
                        data.append({
                            "text": f"{premise} [SEP] {h2}",
                            "aspect": ask_for,
                            "label": 0 if label == 0 else 1
                        })
                except Exception:
                    continue
        save_to_json(data, output / file.with_suffix(".json").name)


def load_mams_dir(folder_path: str, output_folder: str):
    folder = Path(folder_path)
    output = Path(output_folder)
    output.mkdir(parents=True, exist_ok=True)
    for file in tqdm(list(folder.glob("*.xml")), desc="Processing MAMS"):
        data = []
        tree = ET.parse(file)
        root = tree.getroot()
        for sentence in root.findall("sentence"):
            text = sentence.find("text").text
            aspect_terms = sentence.find("aspectTerms")
            if aspect_terms is not None and list(aspect_terms):
                for aspect in aspect_terms:
                    try:
                        term = aspect.attrib["term"]
                        polarity = unify_label(aspect.attrib["polarity"])
                        data.append({"text": text, "aspect": term, "label": polarity})
                    except (KeyError, ValueError):
                        continue
            else:
                data.append({
                    "text": text,
                    "aspect": None,
                    "label": None,
                    "inference_only": True
                })
        save_to_json(data, output / file.with_suffix(".json").name)


def load_sst2_dir(folder_path: str, output_folder: str):
    import pandas as pd

    folder = Path(folder_path)
    output = Path(output_folder)
    output.mkdir(parents=True, exist_ok=True)

    for file in tqdm(list(folder.glob("*.parquet")), desc="Processing SST-2"):
        data = []
        df = pd.read_parquet(file)
        for _, row in df.iterrows():
            text = row.get("sentence")
            label = row.get("label")
            if text is not None and label is not None:
                try:
                    data.append({
                        "text": text,
                        "aspect": None,
                        "label": int(label)
                    })
                except ValueError:
                    continue
        save_to_json(data, output / file.with_suffix(".json").name)



def save_to_json(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} samples to {output_path}")

def preprocess_all_datasets(dataset_dirs: dict, output_root: str):
    """
    dataset_dirs: dict with keys 'semeval', 'ecare', 'mams'
    output_root: directory to save all processed files, grouped by dataset
    """
    output_root = Path(output_root)
    if "semeval" in dataset_dirs:
        load_semeval_dir(dataset_dirs["semeval"], output_root / "semeval")
    if "ecare" in dataset_dirs:
        load_ecare_dir(dataset_dirs["ecare"], output_root / "ecare")
    if "mams" in dataset_dirs:
        load_mams_dir(dataset_dirs["mams"], output_root / "mams")
    if "sst2" in dataset_dirs:
        load_sst2_dir(dataset_dirs["sst2"], output_root / "sst2")

if __name__ == '__main__':
    preprocess_all_datasets(
        dataset_dirs={
            "semeval": "data/SemEval_2014_Task_4/",
            "ecare": "data/e-CARE/data/",
            "mams": "data/MAMS/MAMS-ATSA/raw",
            "sst2": "data/sst-2/"
        },
        output_root="processed/"
    )
