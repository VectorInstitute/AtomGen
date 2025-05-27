"""Run ATOM3D finetuning tasks."""

import argparse
import os
from typing import Any, Callable, Dict

import wandb
from datasets import load_dataset
from transformers import Trainer
from transformers.training_args import TrainingArguments

from atomgen.data.data_collator import DataCollatorForAtomModeling
from atomgen.data.tokenizer import AtomTokenizer
from atomgen.data.utils import (
    compute_metrics_lba,
    compute_metrics_lep,
    compute_metrics_msp,
    compute_metrics_ppi,
    compute_metrics_psr,
    compute_metrics_res,
    compute_metrics_rsr,
    compute_metrics_smp,
)
from atomgen.models.configuration_atomformer import AtomformerConfig
from atomgen.models.modeling_atomformer import AtomFormerForSystemClassification


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run ATOM3D finetuning tasks")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["RES", "PPI", "MSP", "LBA", "LEP", "PSR", "RSR", "SMP"],
        help="ATOM3D task to run",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vector-institute/atomformer-base",
        help="Name or path of the pre-trained model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Path to the output directory",
    )
    parser.add_argument(
        "--num_train_epochs", type=float, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether to use gradient checkpointing",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.001,
        help="Warmup ratio for learning rate scheduler",
    )
    parser.add_argument(
        "--select",
        type=int,
        default=-1,
        help="Number of samples to select from the dataset, -1 for all",
    )

    return parser.parse_args()


def get_task_config(task: str) -> Dict[str, Any]:
    """Get task-specific configuration."""
    task_configs = {
        "SMP": {"num_labels": 20, "problem_type": "regression"},
        "RES": {"num_labels": 20, "problem_type": "multiclass_classification"},
        "MSP": {"num_labels": 1, "problem_type": "classification"},
        "LBA": {"num_labels": 1, "problem_type": "regression"},
        "LEP": {"num_labels": 1, "problem_type": "classification"},
        "PPI": {"num_labels": 1, "problem_type": "classification"},
        "PSR": {"num_labels": 4, "problem_type": "regression"},
        "RSR": {"num_labels": 1, "problem_type": "regression"},
    }
    return task_configs[task]


def get_compute_metrics(task: str) -> Callable[[Any], Dict[str, Any]]:
    """Get task-specific compute_metrics function."""
    task_compute_metrics = {
        "SMP": compute_metrics_smp,
        "RES": compute_metrics_res,
        "MSP": compute_metrics_msp,
        "LBA": compute_metrics_lba,
        "LEP": compute_metrics_lep,
        "PPI": compute_metrics_ppi,
        "PSR": compute_metrics_psr,
        "RSR": compute_metrics_rsr,
    }
    return task_compute_metrics[task]


def run_atom3d(args: argparse.Namespace) -> None:
    """Run ATOM3D finetuning tasks."""
    task_config = get_task_config(args.task)

    # Set up model configuration
    if args.model == "scratch":
        config = AtomformerConfig.from_pretrained(
            "vector-institute/atomformer-base",
            num_labels=task_config["num_labels"],
            gradient_checkpointing=args.gradient_checkpointing
            if args.gradient_checkpointing is not None
            else False,
            problem_type=task_config["problem_type"],
        )
        model = AtomFormerForSystemClassification(config)  # type: ignore[arg-type]
    else:
        config = AtomformerConfig.from_pretrained(
            args.model,
            num_labels=task_config["num_labels"],
            gradient_checkpointing=args.gradient_checkpointing
            if args.gradient_checkpointing is not None
            else False,
            problem_type=task_config["problem_type"],
        )
        model = AtomFormerForSystemClassification.from_pretrained(
            args.model, config=config
        )

    # Load dataset
    dataset = load_dataset(f"vector-institute/atom3d-{args.task.lower()}")
    if args.select != -1:
        for split in dataset:
            dataset[split] = dataset[split].select(
                range(min(args.select, len(dataset[split])))
            )

    # Rename 'label' column to 'labels' if it exists
    if "label" in dataset["train"].features:
        dataset = dataset.rename_column("label", "labels")

    # Set up tokenizer and data collator
    tokenizer = AtomTokenizer(vocab_file="atomgen/data/tokenizer.json")
    data_collator = DataCollatorForAtomModeling(
        tokenizer=tokenizer,
        mam=False,
        coords_perturb=False,
        return_lap_pe=False,
        return_edge_indices=False,
    )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        wandb.login(key=os.environ.get("WANDB_API_KEY"))
        wandb.init(project=args.project, config=vars(args), name=args.name)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, args.task.lower()),
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_dir=os.path.join(args.output_dir, args.task.lower(), "logs"),
        logging_steps=100,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
    )

    # Initialize trainer
    trainer = Trainer(  # type: ignore[no-untyped-call]
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset={"val": dataset["val"], "test": dataset["test"]},
        data_collator=data_collator,
        compute_metrics=get_compute_metrics(args.task),
    )

    # Train the model
    trainer.train()  # type: ignore[attr-defined]

    trainer.evaluate(dataset["test"])  # type: ignore[attr-defined]

    # Save the model
    trainer.save_model(args.output_dir)  # type: ignore[attr-defined]


if __name__ == "__main__":
    args = parse_arguments()
    run_atom3d(args)
