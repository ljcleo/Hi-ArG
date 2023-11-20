from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime
from os import environ
from typing import Any

import torch
import wandb
from collate import DataCollatorForJointInput
from data import AKGDataset
from model import GreaseArgConfig
from model_pretrain import GreaseArgModelForPreTraining
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Subset
from trainer import SUB_LOSS_KEYS, GreaseArgTrainer, GreaseArgTrainingArguments
from transformers import (
    EvalPrediction,
    RobertaConfig,
    RobertaTokenizer,
    is_torch_available,
    set_seed,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names


@dataclass
class TrainArg:
    target: str = None
    mode: int = 0
    comment: int = 0

    no_text_loss: bool = False
    no_match_loss: bool = False
    no_node_loss: bool = False
    no_top_loss: bool = False
    no_edge_loss: bool = False
    no_dir_loss: bool = False
    no_cont_loss: bool = False

    total_batch_size: int = 32
    device_batch_size: int = 8
    gpu_cores: int = 1
    seed: int = 19260817
    no_eval: bool = False
    no_restart: bool = False

    learning_rate: float = 5e-5
    warmup_steps: int = 2500
    total_steps: int = 180000
    logging_steps: int = 500
    eval_steps: int = 10000
    save_steps: int = 10000
    num_loaders: int = 0


def enable_full_determinism(seed: int):
    """
    Helper function for reproducible behavior during distributed training. See
    - https://pytorch.org/docs/stable/notes/randomness.html for pytorch
    """
    # set seed first
    set_seed(seed)

    if is_torch_available():
        # environ['CUDA_LAUNCH_BLOCKING'] = "1"
        environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.use_deterministic_algorithms(True, warn_only=True)

        # Enable CUDNN deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_optimizer(model: GreaseArgModelForPreTraining, args: GreaseArgTrainingArguments) -> AdamW:
    decay_parameters: list[str] = [
        name for name in get_parameter_names(model, ALL_LAYERNORM_LAYERS) if "bias" not in name
    ]

    lm_parameters: list[str] = [
        name for name in get_parameter_names(model, ALL_LAYERNORM_LAYERS) if "roberta" in name
    ]

    lm_learning_rate: float = args.learning_rate
    gnn_learning_rate: float = lm_learning_rate * args.gnn_lr_multiplier

    optimizer_grouped_parameters: list[dict[str, Any]] = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n in decay_parameters and n in lm_parameters
            ],
            "lr": lm_learning_rate,
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n not in decay_parameters and n in lm_parameters
            ],
            "lr": lm_learning_rate,
            "weight_decay": 0.0,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n in decay_parameters and n not in lm_parameters
            ],
            "lr": gnn_learning_rate,
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n not in decay_parameters and n not in lm_parameters
            ],
            "lr": gnn_learning_rate,
            "weight_decay": 0.0,
        },
    ]

    return AdamW(
        optimizer_grouped_parameters,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
    )


def get_scheduler(optimizer: AdamW, args: GreaseArgTrainingArguments) -> LambdaLR:
    warmup: int = args.warmup_steps
    decay: float = warmup**0.5

    def inverse_sqrt(step: int) -> float:
        step += 1

        if step < warmup:
            return step / warmup
        else:
            return decay * step**-0.5

    return LambdaLR(optimizer, inverse_sqrt)


def compute_metrics(pred: EvalPrediction) -> dict[str, float]:
    return {key: loss.mean().item() for key, loss in zip(SUB_LOSS_KEYS, pred.predictions)}


def train(args: TrainArg, is_test: bool) -> None:
    enable_full_determinism(args.seed)
    name: str = f"{args.target}_{args.mode}_{args.comment}"
    tags: list[str] = [f"mode_{args.mode}", f"comment_{args.comment}"]

    if args.no_text_loss:
        print("WARNING: Training without text loss!")
        name = f"{name}_nt"
        tags.append("no_text_loss")

    if args.no_match_loss:
        print("WARNING: Training without match loss!")
        name = f"{name}_nm"
        tags.append("no_match_loss")

    if args.no_node_loss:
        print("WARNING: Training without node loss!")
        name = f"{name}_nn"
        tags.append("no_node_loss")

    if args.no_top_loss:
        print("WARNING: Training without top loss!")
        name = f"{name}_np"
        tags.append("no_top_loss")

    if args.no_edge_loss:
        print("WARNING: Training without edge loss!")
        name = f"{name}_ne"
        tags.append("no_edge_loss")

    if args.no_dir_loss:
        print("WARNING: Training without dir loss!")
        name = f"{name}_nd"
        tags.append("no_dir_loss")

    if args.no_cont_loss:
        print("WARNING: Training without cont loss!")
        name = f"{name}_nc"
        tags.append("no_cont_loss")

    name = f"{name}_{args.seed}"
    tags.append(f"{args.seed}")

    if is_test:
        name = f"{name}_test"
    else:
        wandb.init(
            project="GreaseArG",
            name=f'{name}_{datetime.strftime(datetime.now(), "%m%d%H%M%S")}',
            tags=tags,
        )

    output_dir: str = f"./model/{name}"
    sub_mode: int = 0

    if args.mode >= 3:
        sub_mode = args.mode - 3
        args.mode = 3

    input_text = bool(args.mode & 1)
    input_graph = bool(args.mode & 2)
    freeze_lm_steps: int = 0

    if sub_mode >= 1:
        freeze_lm_steps = 500 if is_test else args.total_steps
        if sub_mode == 1:
            freeze_lm_steps = round(freeze_lm_steps * 0.5)

    print(f"Input text: {input_text}")
    print(f"Input graph: {input_graph}")
    print(f"Freeze LM steps: {freeze_lm_steps}")
    print(f"Comment mode: {args.comment}")

    tokenizer: RobertaTokenizer = RobertaTokenizer.from_pretrained("./tokenizer")
    print("Tokenizer loaded.", flush=True)

    train_dataset = AKGDataset(args.target, "train", args.comment)
    eval_dataset = AKGDataset(args.target, "eval", args.comment)

    if is_test:
        eval_dataset = Subset(eval_dataset, list(range(256)))
    print("Dataset loaded.", flush=True)

    config = GreaseArgConfig.from_roberta_config(
        RobertaConfig.from_pretrained("roberta-base", vocab_size=len(tokenizer)),
        input_text=input_text,
        input_graph=input_graph,
    )

    model = GreaseArgModelForPreTraining(
        config,
        text_loss_weight=1.0 - args.no_text_loss,
        match_loss_weight=1.0 - args.no_match_loss,
        node_loss_weight=1.0 - args.no_node_loss,
        top_loss_weight=1.0 - args.no_top_loss,
        edge_loss_weight=1.0 - args.no_edge_loss,
        dir_loss_weight=1.0 - args.no_dir_loss,
        cont_loss_weight=1.0 - args.no_cont_loss,
    )

    model.grease_arg.load_roberta_pretrained("roberta-base")
    print("Model loaded.", flush=True)

    collator = DataCollatorForJointInput(input_text=input_text, input_graph=input_graph)

    print("Collator loaded.", flush=True)

    if is_test:
        args.total_steps = 500
        args.warmup_steps = 50
        args.logging_steps = 10
        args.save_steps = 250
        args.eval_steps = 50

    training_args = GreaseArgTrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=not args.no_restart,
        evaluation_strategy="no" if args.no_eval else "steps",
        per_device_train_batch_size=args.device_batch_size,
        per_device_eval_batch_size=args.device_batch_size << 1,
        gradient_accumulation_steps=max(
            args.total_batch_size // (args.gpu_cores * args.device_batch_size), 1
        ),
        learning_rate=args.learning_rate,
        max_steps=args.total_steps,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=20,
        seed=args.seed,
        fp16=True,
        eval_steps=args.eval_steps,
        dataloader_num_workers=args.num_loaders,
        remove_unused_columns=False,
        report_to="none" if is_test else "wandb",
        freeze_lm_steps=freeze_lm_steps,
    )

    optimizer: AdamW = get_optimizer(model, training_args)
    scheduler: LambdaLR = get_scheduler(optimizer, training_args)

    trainer = GreaseArgTrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler),
    )

    print("Trainer loaded.", flush=True)

    trainer.train(resume_from_checkpoint=args.no_restart)
    print("Model trained.", flush=True)

    trainer.save_model(output_dir)
    print("Model saved.", flush=True)


def parse_args() -> TrainArg:
    parser = ArgumentParser(description="Pretrain GreaseArG model.")
    parser.add_argument("target", choices=("argsme",), help="Target dataset")

    parser.add_argument(
        "mode",
        type=int,
        choices=(1, 2, 3, 4, 5),
        help="Model mode (1: text, 2: graph, 3: joint, 4: h-frz, 5: f-frz)",
    )

    parser.add_argument(
        "comment",
        type=int,
        choices=(0, 1, 2),
        help="Use commented samples or not (0: no, 1: yes, 2: random)",
    )

    parser.add_argument("--no-text-loss", action="store_true", help="Train without text loss")
    parser.add_argument("--no-match-loss", action="store_true", help="Train without match loss")
    parser.add_argument("--no-node-loss", action="store_true", help="Train without node loss")
    parser.add_argument("--no-top-loss", action="store_true", help="Train without top loss")
    parser.add_argument("--no-edge-loss", action="store_true", help="Train without edge loss")
    parser.add_argument("--no-dir-loss", action="store_true", help="Train without dir loss")
    parser.add_argument("--no-cont-loss", action="store_true", help="Train without cont loss")
    parser.add_argument("-t", "--total-batch-size", type=int, help="Total batch size")
    parser.add_argument("-d", "--device-batch-size", type=int, help="Batch size per device")
    parser.add_argument("-g", "--gpu-cores", type=int, help="Number of GPU cores")
    parser.add_argument("-s", "--seed", type=int, help="Random number generator seed")
    parser.add_argument("-n", "--no-eval", action="store_true", help="Skip evaluation")
    parser.add_argument("-c", "--no-restart", action="store_true", help="Do not restart")
    return parser.parse_args(namespace=TrainArg())


if __name__ == "__main__":
    train(parse_args(), False)
