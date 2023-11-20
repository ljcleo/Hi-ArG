from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from datetime import datetime
from os import environ
from typing import Any

import torch
import wandb
from collate import DataCollatorForJointInput
from data import IAMCESCDataset
from model import GreaseArgConfig
from model_finetune import GreaseArgModelForClassification
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from trainer import IAMCESCTrainer, compute_metrics
from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    TrainingArguments,
    is_torch_available,
    set_seed,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names


@dataclass
class TrainArg:
    mode: int = 0
    pretrain_dataset: str | None = None
    pretrain_comment: int | None = None
    checkpoint: int | None = None
    total_batch_size: int = 128
    device_batch_size: int = 64
    gpu_cores: int = 2
    seed: int = 19260817

    learning_rate: float = 4e-5
    warmup_ratio: int = 0.1
    num_train_epochs: int = 20
    logging_steps: int = 50
    num_loaders: int = 0


def enable_full_determinism(seed: int):
    """
    Helper function for reproducible behavior during distributed training. See
    - https://pytorch.org/docs/stable/notes/randomness.html for pytorch
    """
    set_seed(seed)

    if is_torch_available():
        # environ['CUDA_LAUNCH_BLOCKING'] = "1"
        environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_optimizer(model: GreaseArgModelForClassification, args: TrainingArguments) -> AdamW:
    decay_parameters: list[str] = [
        name for name in get_parameter_names(model, ALL_LAYERNORM_LAYERS) if "bias" not in name
    ]

    optimizer_grouped_parameters: list[dict[str, Any]] = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]

    return AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
    )


def get_scheduler(optimizer: AdamW, args: TrainingArguments) -> LambdaLR:
    warmup: int = args.warmup_steps
    decay: float = warmup**0.5

    def inverse_sqrt(step: int) -> float:
        step += 1

        if step < warmup:
            return step / warmup
        else:
            return decay * step**-0.5

    return LambdaLR(optimizer, inverse_sqrt)


def train(args: TrainArg, is_test: bool) -> None:
    enable_full_determinism(args.seed)
    name: str = str(args.mode)
    sub_name: str = str(args.seed)

    if args.pretrain_dataset is not None:
        assert args.pretrain_comment is not None
        name = "_".join([name, args.pretrain_dataset, str(args.pretrain_comment)])

        if args.checkpoint is not None:
            name = "_".join([name, str(args.checkpoint)])

    if is_test:
        name = f"{name}_test"
    else:
        timestamp: str = datetime.strftime(datetime.now(), "%m%d%H%M")

        wandb.init(
            config=asdict(args),
            project="GreaseArG-IAM-CESC",
            name=f"{name}_{sub_name}_{timestamp}",
        )

    output_dir: str = f"./evaluate/{name}"
    tokenizer: RobertaTokenizer = RobertaTokenizer.from_pretrained("./tokenizer")
    print("Tokenizer loaded.", flush=True)

    train_set: IAMCESCDataset
    dev_set: IAMCESCDataset
    test_set: IAMCESCDataset
    train_set, dev_set, test_set = IAMCESCDataset.load_train_dev_test("iam")
    print("Dataset loaded.", flush=True)

    input_text = bool(min(args.mode, 3) & 1)
    input_graph = bool(min(args.mode, 3) & 2)
    model: GreaseArgModelForClassification

    if args.pretrain_dataset is not None:
        pretrain_name: str = "_".join(
            [str(x) for x in (args.pretrain_dataset, args.mode, args.pretrain_comment, args.seed)]
        )

        pretrain_dir: str = f"./pretrain/{pretrain_name}"
        if args.checkpoint is not None:
            pretrain_dir = f"{pretrain_dir}/checkpoint-{args.checkpoint}"

        model = GreaseArgModelForClassification.from_pretrained(
            pretrain_dir, input_text=input_text, input_graph=input_graph, num_labels=3
        )
    else:
        config = GreaseArgConfig.from_roberta_config(
            RobertaConfig.from_pretrained("roberta-base", vocab_size=len(tokenizer), num_labels=3),
            input_text=input_text,
            input_graph=input_graph,
            num_labels=3,
        )

        model = GreaseArgModelForClassification(config)
        model.grease_arg.load_roberta_pretrained("roberta-base")

    print("Model loaded.", flush=True)

    collator = DataCollatorForJointInput(tokenizer, input_text=input_text, input_graph=input_graph)
    print("Collator loaded.", flush=True)

    if is_test:
        args.num_train_epochs = 1

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=args.device_batch_size,
        per_device_eval_batch_size=args.device_batch_size << 3,
        gradient_accumulation_steps=max(
            args.total_batch_size // (args.gpu_cores * args.device_batch_size), 1
        ),
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=round(args.warmup_ratio * len(train_set) / args.total_batch_size),
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        seed=args.seed,
        fp16=True,
        dataloader_num_workers=args.num_loaders,
        remove_unused_columns=False,
        report_to="none" if is_test else "wandb",
    )

    optimizer: AdamW = get_optimizer(model, training_args)
    scheduler: LambdaLR = get_scheduler(optimizer, training_args)

    trainer = IAMCESCTrainer(
        dump_name=sub_name,
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_set,
        eval_dataset=dev_set,
        test_dataset=test_set,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler),
    )

    print("Trainer loaded.", flush=True)

    trainer.train()
    print("Model trained.", flush=True)

    trainer.save_model(output_dir)
    print("Result saved.", flush=True)


def parse_args() -> TrainArg:
    parser = ArgumentParser(description="Fine-tune GreaseArg model with IAM CESC task.")

    parser.add_argument(
        "mode",
        type=int,
        choices=(1, 2, 3, 4, 5),
        help="Model mode (1: text, 2: graph, 3: both, 4: h-frz, 5: f-frz)",
    )

    parser.add_argument("-s", "--seed", type=int, help="Random number generator seed")
    parser.add_argument("-p", "--pretrain-dataset", type=str, help="Specify pretraining dataset")
    parser.add_argument("-m", "--pretrain-comment", type=int, help="Specify comment style")
    parser.add_argument("-c", "--checkpoint", type=int, help="Specify checkpoint")
    parser.add_argument("-t", "--total-batch-size", type=int, help="Total batch size")
    parser.add_argument("-d", "--device-batch-size", type=int, help="Batch size per device")
    parser.add_argument("-g", "--gpu-cores", type=int, help="Number of GPU cores")
    return parser.parse_args(namespace=TrainArg())


if __name__ == "__main__":
    train(parse_args(), False)
