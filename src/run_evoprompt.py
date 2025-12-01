import os
from promptolution.llms import APILLM
from promptolution.optimizers import EvoPromptGA
from promptolution.predictors import MarkerBasedPredictor
from promptolution.tasks import ClassificationTask
from promptolution.utils import FileOutputCallback, LoggerCallback, TokenCountCallback
from promptolution.utils.prompt_creation import create_prompts_from_task_description
from logging import Logger

logger = Logger(__name__)


def run_evoprompt(df, task_description, model_kwargs, output_dir, token_limit):
    callbacks = [
        LoggerCallback(logger),
        FileOutputCallback(output_dir, file_type="csv"),
        TokenCountCallback(token_limit, "total_tokens"),
    ]

    task = ClassificationTask(
        df,
        task_description=task_description,
        x_column="input",
        y_column="target",
        eval_strategy="sequential_block",
    )
    model_kwargs["call_kwargs"] = {
        "temperature": model_kwargs.pop("temperature"),
        "seed": model_kwargs.pop("seed"),
    }
    llm = APILLM(
        model_id=model_kwargs["name"],
        api_url=model_kwargs["base_url"],
        api_key=model_kwargs["api_key"],
        max_tokens=model_kwargs["max_tokens"],
        call_kwargs=model_kwargs["call_kwargs"],
        call_timeout_s=3000,
        gather_timeout_s=60000.0,
    )
    predictor = MarkerBasedPredictor(llm, classes=None)

    initial_prompts = create_prompts_from_task_description(task_description, llm=llm, n_prompts=10)
    optimizer = EvoPromptGA(
        task=task,
        predictor=predictor,
        meta_llm=llm,
        initial_prompts=initial_prompts,
        callbacks=callbacks,
    )

    best_prompts = optimizer.optimize(n_steps=9999)  # run until token limit reached
    best_prompt = str(best_prompts[0]) # they are sorted by scores!

    logger.info(f"Optimized prompts: {best_prompts}")

    # save prompts to outputdir/best_prompt.txt
    with open(os.path.join(output_dir, "best_prompt.txt"), "w") as f:
        f.write(best_prompt + "\n\n")

    return best_prompts
