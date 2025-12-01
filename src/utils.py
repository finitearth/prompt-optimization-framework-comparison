import random
import re
import numpy as np
from datasets import load_dataset


def get_dataset(dataset, seed=42):
    if dataset == "gsm8k":
        df = load_dataset("openai/gsm8k", "main")["train"].to_pandas()
        df["input"] = df["question"]
        df["target"] = df["answer"].str.extract(r"#### (.*)")
        train_df = df.sample(n=500, random_state=seed)
        test_df = df.drop(train_df.index).sample(n=300, random_state=seed)
    elif dataset == "sst5":
        df = load_dataset("SetFit/sst5")["train"].to_pandas()
        df["input"] = df["text"]
        df["target"] = df["label_text"]
        train_df = df.sample(500, random_state=seed)
        test_df = df.drop(train_df.index).sample(300, random_state=seed)
    else:
        raise ValueError("unknown dataset!")
    train_df = train_df.reset_index(drop=True)
    train_df["target"] = train_df["target"].str.lower()
    test_df["target"] = test_df["target"].str.lower()
    return train_df, test_df


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def extract_prediction(pred, dspy_fix=False):
    if pred is None:
        return ""
    if dspy_fix:
        # dspy prompts the LLM to add certain tags, that internally gets removed, which is not the case when we evaluate in langchain
        pred = pred.replace("[[ ## target ## ]]", "")
        pred = pred.replace("[[ ## completed ## ]]", "")
    match = re.search(r"<final_answer>(.*?)</final_answer>", pred, re.DOTALL | re.IGNORECASE)
    extracted = match.group(1).strip() if match else pred.strip()
    extracted = extracted.lower()
    return extracted
