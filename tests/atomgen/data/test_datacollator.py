"""Tests for DataCollator."""
import torch

from atomgen.data.data_collator import DataCollatorForAtomModeling
from atomgen.data.tokenizer import AtomTokenizer


def test_data_collator():
    """Test DataCollatorForAtomModeling."""
    tokenizer = AtomTokenizer(vocab_file="atomgen/data/tokenizer.json")
    data_collator = DataCollatorForAtomModeling(
        tokenizer=tokenizer,
        mam=False,
        coords_perturb=False,
        causal=False,
        return_edge_indices=False,
        pad=True,
    )
    size = torch.randint(4, 16, (10,)).tolist()
    dataset = [
        {
            "input_ids": torch.randint(0, 123, (size[i],)).tolist(),
            "coords": torch.randint(0, 123, (size[i], 3)).tolist(),
        }
        for i in range(10)
    ]
    batch = data_collator(dataset)
    assert len(batch["input_ids"]) == 10
