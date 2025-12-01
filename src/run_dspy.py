import dspy
import logging
import os
import time
import pandas as pd
from dspy.teleprompt import GEPA

from src.utils import extract_prediction


logger = logging.getLogger(__name__)


class TokenLimitExceeded(BaseException):
    pass

class GlobalTokenTracker:
    def __init__(self):
        self.total = 0
        self.last = 0

    def add_usage(self, _, record):
        tokens = record["total_tokens"]
        self.last = tokens
        self.total += tokens

    def reset(self):
        self.total = 0
        self.last = 0

global_token_tracker = GlobalTokenTracker()

class TaskSignature(dspy.Signature):
    input = dspy.InputField()
    target = dspy.OutputField()


class ExperimentLM(dspy.LM):
    def __init__(
        self, model: str, limit: int, api_key: str, base_url: str, log_path: str, **kwargs
    ):
        # Normalize model name for OpenAI adapter
        if not model.startswith("openai/"):
            model = f"openai/{model}"

        super().__init__(model=model, api_key=api_key, api_base=base_url, **kwargs)
        self.token_limit = limit
        self.log_path = log_path

        self.log_buffer = []

    def log_prompt(self, prompt, response, token_count):
        self.log_buffer.append(
            {
                "timestamp": time.time(),
                "prompt": prompt,
                "response": response,
                "tokens_used": token_count,
            }
        )

    def flush_logs(self):
        df = pd.DataFrame(self.log_buffer)
        df.to_parquet(self.log_path, index=False)

    def __call__(self, prompt=None, **kwargs):
        global global_token_tracker

        if global_token_tracker.total >= self.token_limit:
            logger.error("Token limit exceeded in ExperimentLM.")
            raise TokenLimitExceeded(f"Hard limit of {self.token_limit} tokens reached.")

        result = super().__call__(prompt, **kwargs)
        try:
            system_prompt = self.history[-1]["messages"][0]["content"]
            user_prompt = self.history[-1]["messages"][1]["content"]
            prompt = f"[[SYSTEMPROMPT]]{system_prompt}[[USERPROMPT]]{user_prompt}"
        except Exception:
            prompt = "UNKNOWN"
            logger.warning(f"Could not extract prompt from history for logging.\nHistory: {self.history[-1]}")
        self.log_prompt(prompt, str(result), global_token_tracker.total)

        if len(self.history) % 100 == 0:
            logger.info(f"Token Usage: {global_token_tracker.total}/{self.token_limit}")

        return result


class TaskModule(dspy.Module):
    def __init__(self, task_description: str):
        super().__init__()
        self.signature_class = TaskSignature
        self.signature_class.__doc__ = task_description

        self.prog = dspy.Predict(self.signature_class)

    def forward(self, input):
        return self.prog(input=input)



def metric(gold, pred, *_, **__):
    try:
        raw_text = pred.target
        pred = extract_prediction(raw_text)

        score = 1 if pred == str(gold.target) else 0
    except Exception as e:
        logger.error(f"Error during metric calculation: {e}")
        score = 0

    return score


def run_dspy(df, task_description, model_kwargs, output_dir, token_limit):
    dspy.settings.usage_tracker = global_token_tracker
    call_log = os.path.join(output_dir, "call_history.parquet")
    df_train = df.sample(frac=0.7, random_state=42)
    df_val = df.drop(df_train.index)
    trainset = []
    for i, row in df_train.iterrows():
        trainset.append(
            dspy.Example(
                id=str(i),
                input=str(row["input"]),
                target=str(row["target"]),
            ).with_inputs("input")
        )

    valset = []
    for i, row in df_val.iterrows():
        valset.append(
            dspy.Example(
                id=str(i),
                input=str(row["input"]),
                target=str(row["target"]),
            ).with_inputs("input")
        )

    lm = ExperimentLM(
        model=model_kwargs["name"],
        limit=token_limit,
        api_key=model_kwargs["api_key"],
        base_url=model_kwargs["base_url"],
        log_path=call_log,
        temperature=model_kwargs["temperature"],
        seed=model_kwargs["seed"],
    )
    dspy.settings.configure(lm=lm)
    program = TaskModule(task_description=task_description)
    teleprompter = GEPA(
        metric=metric,
        reflection_lm=lm,
        seed=model_kwargs["seed"],
        log_dir=output_dir+"/teleprompter_logs",
        max_metric_calls=999 # will be stopped by token limit
    )

    try:
        best_program = teleprompter.compile(
            program,
            trainset=trainset,
            valset=valset,
        )
        final_prompts = best_program.signature_class.instructions

    except TokenLimitExceeded:
        logger.warning("Token limit reached! Stopping.")
        final_prompts = lm.log_buffer[-1]["prompt"]


    lm.flush_logs()

    logger.info(f"optimization finished, extracting - best prompt:\n{final_prompts}")
