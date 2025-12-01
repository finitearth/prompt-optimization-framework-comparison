from dataclasses import dataclass

import adalflow as adal
from adalflow.components.model_client.openai_client import OpenAIClient
from adalflow.eval.answer_match_acc import AnswerMatchAcc
from adalflow.optim.text_grad.text_loss_with_eval_fn import EvalFnToTextLoss
from src.utils import extract_prediction

import pandas as pd
from logging import Logger

logger = Logger(__name__)


class TokenLimitExceeded(BaseException):
    pass


@dataclass
class DataPoint:
    id: str
    input: str
    target: str


class CustomClient(OpenAIClient):
    def __init__(self, token_limit, log_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_limit = token_limit
        self.log_path = log_path
        
        self.used_tokens = 0
        self.log_buffer = []

    def call(self, api_kwargs, *_, **__):
        if self.used_tokens > self.token_limit:
            raise TokenLimitExceeded(f"Token limit of {self.token_limit} exceeded.")

        api_kwargs["model"] = api_kwargs.pop("name")
        api_kwargs.pop("api_key")
        api_kwargs.pop("base_url")
        api_kwargs.pop("seed")

        messages = self._convert_llm_inputs_to_messages(input=api_kwargs["input"])
        if len(messages) == 1:
            # unfortunately there is seemingly a bug, that the prompts provided to the optimizer and backward engine llm contain  <START_OF_USER> and <END_OF_USER> which produces a bug - so here is a whacky workaround
            api_kwargs["input"] = (
                api_kwargs["input"]
                .replace("<START_OF_USER>", "<START_OF_USER_PROMPT>")
                .replace("<END_OF_USER>", "<END_OF_USER_PROMPT>")
                .replace("<START_OF_USER_MESSAGE>", "<START_OF_USER_PROMPT>")
                .replace("<END_OF_USER_MESSAGE>", "<END_OF_USER_PROMPT>")
            )

            messages = self._convert_llm_inputs_to_messages(input=api_kwargs["input"])
        else:
            self.log_prompt(api_kwargs["input"], self.used_tokens) # only log prompt if its not meta-prompt!
        api_kwargs["input"] = messages
        completion = self.sync_client.responses.create(**api_kwargs)
        self.used_tokens += completion.usage.total_tokens

        if self.used_tokens > self.token_limit:
            raise TokenLimitExceeded(f"Token limit of {self.token_limit} exceeded.")

        return completion

    def log_prompt(self, prompt, token_used):
        self.log_buffer.append(
            {"prompt": prompt, "token_used": token_used, "cumulative_tokens": self.used_tokens}
        )

    def flush_logs(self):
        df = pd.DataFrame(self.log_buffer)
        df.to_parquet(self.log_path, index=False)


class SimpleInstructionTask(adal.Component):
    def __init__(self, model_client, model_kwargs, task_description):
        super().__init__()

        self.system_prompt = adal.Parameter(
            data=task_description,
            role_desc="Task instruction to be optimized",
            param_type=adal.ParameterType.PROMPT,
            requires_opt=True,
        )

        prompt_kwargs = {"task_desc_str": self.system_prompt}

        self.llm = adal.Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            prompt_kwargs=prompt_kwargs,
            use_cache=False,
        )

    def call(self, input_text, id=None):
        prompt_kwargs = {
            "input_str": adal.Parameter(
                data=input_text,
                requires_opt=False,
                role_desc="Input text",
            )
        }
        return self.llm(prompt_kwargs=prompt_kwargs, id=id)


class CustomTask(adal.AdalComponent):
    def __init__(
        self,
        model_client,
        model_kwargs,
        task_description,
        model_config,
    ):
        task = SimpleInstructionTask(model_client, model_kwargs, task_description)
        eval_fn = AnswerMatchAcc(type="exact_match").compute_single_item
        loss_fn = EvalFnToTextLoss(
            eval_fn=eval_fn,
            eval_fn_desc="exact_match: 1 if str(y) == str(y_gt) else 0", #Extracting from <final_answer> tags and checking for exact match! ?
        )

        super().__init__(
            task=task,
            eval_fn=eval_fn,
            loss_fn=loss_fn,
            text_optimizer_model_config=model_config,
            backward_engine_model_config=model_config,
            use_loss_eval_fn=True,
        )

    def prepare_task(self, sample):
        return self.task.call, {
            "input_text": sample.input,
            "id": sample.id,
        }

    def prepare_loss(self, sample, y_pred, *args, **kwargs):
        pred_text = y_pred.data.raw_response
        y_pred.eval_input = extract_prediction(pred_text)

        y_gt = adal.Parameter(
            name="y_gt",
            data=sample.target,
            eval_input=sample.target,
            requires_opt=False,
            role_desc="Ground Truth",
        )

        return self.loss_fn, {
            "kwargs": {"y": y_pred, "y_gt": y_gt},
            "id": sample.id,
        }

    def prepare_eval(self, sample, y_pred, *args, **kwargs):
        pred_text = y_pred.data
        y_extracted = extract_prediction(pred_text)

        return self.eval_fn, {"y": y_extracted, "y_gt": sample.target}


def run_adalflow(df, task_description, model_kwargs, output_dir, token_limit):
    # split 70% train, 20% val, 10% test
    df_train = df.sample(frac=0.7, random_state=42)
    df_temp = df.drop(df_train.index)
    df_val = df_temp.sample(frac=2 / 3, random_state=42)
    df_test = df_temp.drop(df_val.index)
    trainset = []
    for i, row in df_train.iterrows():
        trainset.append(
            DataPoint(
                id=str(i),
                input=str(row["input"]),
                target=str(row["target"]),
            )
        )

    valset = []
    for i, row in df_val.iterrows():
        valset.append(
            DataPoint(
                id=str(i),
                input=str(row["input"]),
                target=str(row["target"]),
            )
        )

    testset = []
    for i, row in df_test.iterrows():
        testset.append(
            DataPoint(
                id=str(i),
                input=str(row["input"]),
                target=str(row["target"]),
            )
        )
    client = CustomClient(
        token_limit=token_limit,
        log_path=f"{output_dir}/all_prompts.parquet",
        api_key=model_kwargs["api_key"],
        base_url=model_kwargs["base_url"],
        input_type="messages",  # required, otherwhise outputs jibberish!
    )

    model_config = {
        "model_client": client,
        "model_kwargs": model_kwargs,
    }
    
    adal_task = CustomTask(
        model_client=client,
        model_kwargs=model_kwargs,
        task_description=task_description,
        model_config=model_config,
    )

    trainer = adal.Trainer(
        adaltask=adal_task,
    )
    try:
        trainer.fit(
            train_dataset=trainset,
            val_dataset=valset,
            test_dataset=testset,
        )
    except TokenLimitExceeded as e:
        logger.warning(f"Stopping optimization: {e}")
        
    client.flush_logs()
