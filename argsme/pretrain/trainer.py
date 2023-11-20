from dataclasses import dataclass

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from transformers import Trainer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments

SUB_LOSS_KEYS: tuple[str, ...] = (
    "text_loss",
    "match_loss",
    "node_loss",
    "top_loss",
    "edge_loss",
    "dir_loss",
    "cont_loss",
)


@dataclass
class GreaseArgTrainingArguments(TrainingArguments):
    gnn_lr_multiplier: float = 2.0
    freeze_lm_steps: int = 0


class GreaseArgTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_callback(GreaseArgFreezingCallback(self))
        self._detailed_loss: dict[str, torch.Tensor] = {k: torch.tensor(0.0) for k in SUB_LOSS_KEYS}

    def training_step(
        self, model: nn.Module, inputs: dict[str, torch.Tensor | Batch]
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

            for k in self._detailed_loss:
                if k in outputs:
                    section = outputs[k].detach()
                else:
                    section = torch.tensor(0.0).to(loss)

                v = self._detailed_loss[k].to(section)
                self._detailed_loss[k] = v + section

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.detach()

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Batch],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, tuple[torch.Tensor] | None, tuple[torch.Tensor] | None]:
        inputs = self._prepare_inputs(inputs)
        labels = ()

        if "input_ids_labels" in inputs:
            labels = labels + (inputs["input_ids_labels"].detach(),)
        if "match_labels" in inputs:
            labels = labels + (inputs["match_labels"].detach(),)

        if "good_graphs" in inputs:
            graphs = inputs["good_graphs"]
            labels = labels + (graphs.y.detach(), graphs.edge_y.detach(), graphs.edge_dir.detach())

        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

            loss = loss.mean().detach()

            detailed_loss = tuple(
                outputs[k].detach() if k in outputs else torch.tensor(0).to(loss)
                for k in SUB_LOSS_KEYS
            )

        if prediction_loss_only:
            return (loss, None, None)
        return (loss, detailed_loss, labels)

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            logs: dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            steps = self.state.global_step - self._globalstep_last_logged
            logs["loss"] = round(tr_loss_scalar / steps, 4)
            logs["learning_rate"] = self._get_learning_rate()

            for k in self._detailed_loss:
                v = self._detailed_loss[k]
                scalar = self._nested_gather(v).mean().item()
                self._detailed_loss[k] -= v

                logs[k] = round(scalar / steps / self.args.gradient_accumulation_steps, 4)

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)

            self._report_to_hp_search(trial, self.state.global_step, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)


class GreaseArgFreezingCallback(TrainerCallback):
    def __init__(self, trainer: GreaseArgTrainer) -> None:
        super().__init__()
        self.trainer = trainer
        self.requires_grad: bool = True

    def on_step_begin(
        self,
        args: GreaseArgTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        requires_grad: bool = state.global_step >= args.freeze_lm_steps
        self._set_lm_requires_grad(self.trainer.model, requires_grad)

    def on_save(
        self,
        args: GreaseArgTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self._set_lm_requires_grad(self.trainer.model, True)

    def _set_lm_requires_grad(self, model: nn.Module, state: bool) -> None:
        if self.requires_grad == state:
            return

        self.requires_grad = state

        for name, param in model.named_parameters():
            if ".roberta." in name:
                param.requires_grad = state
