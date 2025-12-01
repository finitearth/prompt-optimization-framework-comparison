# example usage: uv run -m src.run_evaluation --model_config configs/model_config.yaml --task_config configs/datasets/sst5.yaml --system_prompt "You are a helpful Assistant." --prompt_template 'You are a sentiment analyzer focused on identifying the emotional polarity of text. Read each movie review carefully and assign it to the appropriate sentiment category: very negative, negative, neutral, positive, or very positive. <final_answer>...</final_answer>\n{input}'

import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime

from src.utils import get_dataset, seed_everything
import pandas as pd
from langchain_openai import ChatOpenAI

from src.utils import get_dataset, seed_everything, extract_prediction

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description="Evaluate prompts on test set")

parser.add_argument("--model_config", type=str, required=True)
parser.add_argument("--task_config", type=str, required=True)
parser.add_argument("--system_prompt", type=str, required=True)
parser.add_argument("--prompt_template", type=str, required=True)
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def evaluate_prompt(
    subset,
    system_prompt,
    prompt_template,
    model_kwargs,
    seed,
):
    seed_everything(seed)
    _, test_df = get_dataset(subset, seed=seed)

    print(f"Loaded subset '{subset}' → test size: {len(test_df)}")

    llm = ChatOpenAI(
        model=model_kwargs["name"],
        base_url=model_kwargs["base_url"],
        api_key=model_kwargs["api_key"],
        temperature=model_kwargs["temperature"],
        max_tokens=model_kwargs["max_tokens"],
        seed=seed
    )

    all_messages = []
    golds = []

    for _, row in test_df.iterrows():
        golds.append(str(row["target"]).strip())
        query = prompt_template.format(input=row["input"])
        all_messages.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ])
    
    responses = llm.batch(all_messages)

    
    preds = [extract_prediction(r.content, dspy_fix=True) for r in responses]

    df = pd.DataFrame({
        "input": test_df["input"],
        "target": golds,
        "prediction_raw": preds,
    })

    df["correct"] = (
        df["target"].str.strip() == df["prediction_raw"].str.strip()
    ).astype(int)

    accuracy = df["correct"].mean()
    print(f"\nFinal accuracy on test set: {accuracy:.4f}")

    return df, accuracy


def main():
    model_kwargs = load_yaml(args.model_config)
    task_config = load_yaml(args.task_config)

    logger.info(f"Running evaluation\n" f"model={model_kwargs}\n" f"task={task_config}\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = Path(f"results_eval/{task_config['name']}/{model_kwargs['name']}/run_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Seeding
    seed_everything(args.seed)

    # Load API key
    model_kwargs["api_key"] = open("token.txt", "r").read().strip()

    logger.info("Loading dataset…")
    train_df, test_df = get_dataset(task_config["name"], seed=args.seed)

    logger.info(f"Test split size: {len(test_df)}")

    logger.info("Running evaluation…")

    # Evaluate
    df, acc = evaluate_prompt(
        system_prompt=args.system_prompt,
        prompt_template=args.prompt_template,
        subset=task_config["name"],
        model_kwargs=model_kwargs,
        seed=args.seed,
    )

    # Save results
    df.to_csv(output_dir / "predictions.csv", index=False)

    with open(output_dir / "metrics.txt", "w") as f:
        f.write(f"accuracy: {acc}\n")

    with open(output_dir / "system_prompt.txt", "w") as f:
        f.write(args.system_prompt)

    with open(output_dir / "prompt_template.txt", "w") as f:
        f.write(args.prompt_template)

    logger.info(f"Evaluation finished. Accuracy = {acc:.4f}")
    logger.info(f"Saved predictions to {output_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
