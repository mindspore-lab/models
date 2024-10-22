from datasets import load_dataset

# 创建数据集
dataset = load_dataset("./path/to/data", split="dev")
ds = dataset.train_test_split(test_size=0.1, seed=1)

# 转化数据集并保存
ds_train = ds["train"].rename_column("positive", "pos").rename_column("negative", "neg").filter(lambda x: len(x["pos"]) > 0 and len(x["neg"]) > 0)
ds_train.to_json("train.jsonl", force_ascii=False)
