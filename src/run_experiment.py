# Example usage: uv run -m src.run_experiment --optimizer capo --task_config configs/datasets/gsm8k.yaml --token_limit 1000000


import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime

from src.run_capo import run_capo
from src.run_opro import run_opro
from src.run_evoprompt import run_evoprompt
from src.run_adalflow import run_adalflow
from src.run_dspy import run_dspy
from src.utils import get_dataset, seed_everything
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description="Run Prompt Optimization Experiments")

parser.add_argument("--optimizer", type=str, required=True)
parser.add_argument("--model_config", type=str, default="configs/model_config.yaml")
parser.add_argument("--task_config", type=str, required=True)
parser.add_argument("--token_limit", type=int, default=1_000_000)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

optim_funcs = {
    "capo": run_capo,
    "evoprompt": run_evoprompt,
    "opro": run_opro,
    "dspy": run_dspy,
    "adalflow": run_adalflow,
}


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    model_kwargs = load_yaml(args.model_config)
    task_config = load_yaml(args.task_config)

    logger.info(
        f"setting up experiment with optimizer={args.optimizer}\nmodel={model_kwargs}\ntask={task_config}\ntoken_limit={args.token_limit}"
    )

    optim_func = optim_funcs[args.optimizer]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = Path(
        f"results/{task_config['name']}/{args.optimizer}/{model_kwargs['name']}/run_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    model_kwargs["api_key"] = open("token.txt", "r").read().strip()

    train_df, test_df = get_dataset(task_config["name"], seed=args.seed)

    os.makedirs(output_dir, exist_ok=True)
    logger.info("Starting experiment...")

    optim_func(
        df=train_df,
        task_description=task_config["task_description"],
        model_kwargs=model_kwargs,
        output_dir=str(output_dir),
        token_limit=args.token_limit,
    )

    logger.info("Experiment finished successfully.")


if __name__ == "__main__":
    main()
