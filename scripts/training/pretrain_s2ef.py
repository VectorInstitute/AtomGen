import os
import wandb
import torch
import argparse
import importlib

from transformers import TrainingArguments, Trainer
from datasets import load_from_disk
import numpy as np

from atomgen.data.tokenizer import AtomTokenizer
from atomgen.data.data_collator import DataCollatorForAtomModeling
from atomgen.models.modeling_atomformer import Structure2EnergyAndForces
from atomgen.models.configuration_atomformer import AtomformerConfig


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="seed for the training run")
    parser.add_argument("--project", type=str, default="AtomGen", help="Name of the wandb project")
    parser.add_argument("--name", type=str, default="s2ef_15m_train_base_10epochs", help="Name of the wandb run")
    parser.add_argument("--output_dir", type=str, default=f"./checkpoint", help="Path to the output directory")
    parser.add_argument("--dataset_dir", type=str, default="./s2ef_15m", help="Path to the dataset directory")
    parser.add_argument("--weights_dir", type=str, default="none", help="Path to pre-trained model weights")
    parser.add_argument("--model_config", type=str, default="atomgen/models/configs/atomformer-base.json", help="Path to the model config")
    parser.add_argument("--tokenizer_json", type=str, default="atomgen/data/tokenizer.json", help="Path to the tokenizer")
    parser.add_argument("--micro_batch_size", type=int, default=8, help="Micro batch size")
    parser.add_argument("--macro_batch_size", type=int, default=128, help="Macro batch size")
    parser.add_argument("--max_steps", type=int, default=-1, help="Maximum number of training steps")
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, help="Whether to use gradient checkpointing")
    parser.add_argument("--num_train_epochs", type=float, default=10, help="Number of training epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.001, help="Warmup ratio for learning rate scheduler")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Type of learning rate scheduler")
    parser.add_argument("--weight_decay", type=float, default=1.0e-2, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=5.0, help="Maximum gradient norm")
    parser.add_argument("--eval_accum_steps", type=int, default=1000, help="Number of eval steps to accumulate")
    parser.add_argument("--save_steps", type=int, default=3000, help="Number of steps to save the model")
    parser.add_argument("--log_steps", type=int, default=150, help="Number of steps to log the training")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--num_cpus", type=int, default=-1, help="Number of cpus available")
    parser.add_argument("--save_total_limit", type=int, default=1, help="Total number of checkpoints to keep")
    parser.add_argument("--dataloader_num_workers", type=int, default=-1, help="Number of dataloader workers")
    parser.add_argument("--group_by_length", action=argparse.BooleanOptionalAction, help="Whether to group samples by length")
    parser.add_argument("--length_column_name", type=str, default="length", help="Name of the length column")
    parser.add_argument('--checkpoint-exists', action=argparse.BooleanOptionalAction)
    return parser.parse_args()

def train(args):
    args.output_dir = os.path.join(args.output_dir, args.name)

    torch.manual_seed(args.seed)
    if args.num_cpus == -1:
        args.num_cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()

    dataset = load_from_disk(args.dataset_dir)

    if args.weights_dir != "none":
        model = Structure2EnergyAndForces.from_pretrained(args.weights_dir)
        print(f"PRE-TRAINED WEIGHT LOADED FROM {args.weights_dir}")
    else:
        config = AtomformerConfig.from_json_file(args.model_config)
        config.gradient_checkpointing = args.gradient_checkpointing if args.gradient_checkpointing else False
        model = Structure2EnergyAndForces(config)

    tokenizer = AtomTokenizer(vocab_file=args.tokenizer_json)
    data_collator = DataCollatorForAtomModeling(tokenizer=tokenizer, mam=False, coords_perturb=False, causal=False, return_lap_pe=False, return_edge_indices=False)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        wandb.login(key=os.environ["WANDB_API_KEY"])
        wandb.init(project=args.project, config=vars(args), name=args.name)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        save_strategy="steps",
        save_steps=args.save_steps//(args.macro_batch_size//args.micro_batch_size),
        warmup_ratio=args.warmup_ratio,
        per_device_eval_batch_size=args.micro_batch_size,
        per_device_train_batch_size=args.micro_batch_size,
        eval_accumulation_steps=args.eval_accum_steps//(args.macro_batch_size//args.micro_batch_size),
        gradient_accumulation_steps=args.macro_batch_size//args.micro_batch_size,
        label_names=["formation_energy", "forces", "total_energy"],
        save_total_limit=args.save_total_limit,
        logging_steps=args.log_steps//(args.macro_batch_size//args.micro_batch_size),
        report_to="wandb",
        log_on_each_node=False,
        dataloader_num_workers=args.dataloader_num_workers if args.dataloader_num_workers != -1 else args.num_cpus,
        group_by_length=args.group_by_length,
        length_column_name=args.length_column_name,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=args.checkpoint_exists)

    model.save_pretrained(args.output_dir)

if __name__ == "__main__":
    args = parse_arguments()
    train(args)