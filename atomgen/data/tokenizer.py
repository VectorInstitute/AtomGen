"""tokenization module for atom modeling."""

import collections
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import (
    BatchEncoding,
    EncodedInput,
    Mapping,
    PaddingStrategy,
    TensorType,
)


VOCAB_FILES_NAMES: Dict[str, str] = {"vocab_file": "tokenizer.json"}


class AtomTokenizer(PreTrainedTokenizer):  # type: ignore[misc]
    """
    Tokenizer for atomistic data.

    Args:
        vocab_file: The path to the vocabulary file.
        pad_token: The padding token.
        mask_token: The mask token.
        bos_token: The beginning of system token.
        eos_token: The end of system token.
        cls_token: The classification token.
        kwargs: Additional keyword arguments.

    """

    def __init__(
        self,
        vocab_file: str,
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
        cls_token: str = "<graph>",
        **kwargs: Dict[str, Union[bool, str, PaddingStrategy]],
    ) -> None:
        self.vocab: Dict[str, int] = self.load_vocab(vocab_file)
        self.ids_to_tokens: Dict[int, str] = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()]
        )

        super().__init__(
            pad_token=pad_token,
            mask_token=mask_token,
            bos_token=bos_token,
            eos_token=eos_token,
            cls_token=cls_token,
            **kwargs,
        )

    @staticmethod
    def load_vocab(vocab_file: str) -> Dict[str, int]:
        """Load the vocabulary from a json file."""
        with open(vocab_file, "r") as f:
            vocab = json.load(f)
            if not isinstance(vocab, dict):
                raise ValueError(
                    "The vocabulary file is not a json file or is not formatted correctly."
                )
        return vocab

    def _tokenize(self, text: str) -> List[str]:
        """Split the text into chemical symbols."""
        return re.findall("[A-Z][a-z]*", text)

    def _convert_token_to_id(self, token: str) -> int:
        """Convert the chemical symbols to atomic numbers."""
        return self.vocab[token]

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens[index]

    def get_vocab(self) -> Dict[str, int]:
        """Get the vocabulary."""
        return self.vocab

    def get_vocab_size(self) -> int:
        """Get the size of the vocabulary."""
        return len(self.vocab)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert the list of chemical symbol tokens to a concatenated string."""
        return "".join(tokens)

    def pad(
        self,
        encoded_inputs: Union[
            BatchEncoding,
            List[BatchEncoding],
            Dict[str, EncodedInput],
            Dict[str, List[EncodedInput]],
            List[Dict[str, EncodedInput]],
        ],
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        verbose: bool = True,
    ) -> BatchEncoding:
        """Pad the input data."""
        if isinstance(encoded_inputs, list):
            if isinstance(encoded_inputs[0], Mapping):
                if any(
                    key.startswith("coords") or key.endswith("coords")
                    for key in encoded_inputs[0]
                ):
                    encoded_inputs = self.pad_coords(
                        encoded_inputs,
                        max_length=max_length,
                        pad_to_multiple_of=pad_to_multiple_of,
                    )
                if any(
                    key.startswith("forces") or key.endswith("forces")
                    for key in encoded_inputs[0]
                ):
                    encoded_inputs = self.pad_forces(
                        encoded_inputs,
                        max_length=max_length,
                        pad_to_multiple_of=pad_to_multiple_of,
                    )
                if any(
                    key.startswith("fixed") or key.endswith("fixed")
                    for key in encoded_inputs[0]
                ):
                    encoded_inputs = self.pad_fixed(
                        encoded_inputs,
                        max_length=max_length,
                        pad_to_multiple_of=pad_to_multiple_of,
                    )
        elif isinstance(encoded_inputs, Mapping):
            if any("coords" in key for key in encoded_inputs):
                encoded_inputs = self.pad_coords(
                    encoded_inputs,
                    max_length=max_length,
                    pad_to_multiple_of=pad_to_multiple_of,
                )
            if any("fixed" in key for key in encoded_inputs):
                encoded_inputs = self.pad_fixed(
                    encoded_inputs,
                    max_length=max_length,
                    pad_to_multiple_of=pad_to_multiple_of,
                )

        return super().pad(
            encoded_inputs=encoded_inputs,
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            return_tensors=return_tensors,
            verbose=verbose,
        )

    def pad_coords(
        self,
        batch: Union[Mapping, List[Mapping]],
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> Union[Mapping, List[Mapping]]:
        """Pad the coordinates to the same length."""
        if isinstance(batch, Mapping):
            coord_keys = [
                key
                for key in batch
                if key.startswith("coords") or key.endswith("coords")
            ]
        elif isinstance(batch, list):
            coord_keys = [
                key
                for key in batch[0]
                if key.startswith("coords") or key.endswith("coords")
            ]
        for key in coord_keys:
            if isinstance(batch, Mapping):
                coords = batch[key]
            elif isinstance(batch, list):
                coords = [sample[key] for sample in batch]
            max_length = (
                max([len(c) for c in coords]) if max_length is None else max_length
            )
            if pad_to_multiple_of is not None and max_length % pad_to_multiple_of != 0:
                max_length = (
                    (max_length // pad_to_multiple_of) + 1
                ) * pad_to_multiple_of
            for c in coords:
                c.extend([[0.0, 0.0, 0.0]] * (max_length - len(c)))
            if isinstance(batch, list):
                for i, sample in enumerate(batch):
                    sample[key] = coords[i]
        return batch

    def pad_forces(
        self,
        batch: Union[Mapping, List[Mapping]],
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> Union[Mapping, List[Mapping]]:
        """Pad the forces to the same length."""
        if isinstance(batch, Mapping):
            force_keys = [
                key
                for key in batch
                if key.startswith("forces") or key.endswith("forces")
            ]
        elif isinstance(batch, list):
            force_keys = [
                key
                for key in batch[0]
                if key.startswith("forces") or key.endswith("forces")
            ]
        for key in force_keys:
            if isinstance(batch, Mapping):
                forces = batch[key]
            elif isinstance(batch, list):
                forces = [sample[key] for sample in batch]
            max_length = (
                max([len(c) for c in forces]) if max_length is None else max_length
            )
            if pad_to_multiple_of is not None and max_length % pad_to_multiple_of != 0:
                max_length = (
                    (max_length // pad_to_multiple_of) + 1
                ) * pad_to_multiple_of
            for f in forces:
                f.extend([[0.0, 0.0, 0.0]] * (max_length - len(f)))
            if isinstance(batch, list):
                for i, sample in enumerate(batch):
                    sample[key] = forces[i]
        return batch

    def pad_fixed(
        self,
        batch: Union[Mapping, List[Mapping]],
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> Union[Mapping, List[Mapping]]:
        """Pad the fixed mask to the same length."""
        if isinstance(batch, Mapping):
            fixed_keys = [
                key for key in batch if key.startswith("fixed") or key.endswith("fixed")
            ]
        elif isinstance(batch, list):
            fixed_keys = [
                key
                for key in batch[0]
                if key.startswith("fixed") or key.endswith("fixed")
            ]
        for key in fixed_keys:
            if isinstance(batch, Mapping):
                fixed = batch[key]
            elif isinstance(batch, list):
                fixed = [sample[key] for sample in batch]
            max_length = (
                max([len(c) for c in fixed]) if max_length is None else max_length
            )
            if pad_to_multiple_of is not None and max_length % pad_to_multiple_of != 0:
                max_length = (
                    (max_length // pad_to_multiple_of) + 1
                ) * pad_to_multiple_of
            for f in fixed:
                f.extend([True] * (max_length - len(f)))
            if isinstance(batch, list):
                for i, sample in enumerate(batch):
                    sample[key] = fixed[i]
        return batch

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        """Save the vocabulary to a json file."""
        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "")
            + VOCAB_FILES_NAMES["vocab_file"],
        )
        with open(vocab_file, "w") as f:
            json.dump(self.vocab, f)
        return (vocab_file,)

    @classmethod
    def from_pretrained(cls, *inputs: Any, **kwargs: Any) -> Any:
        """Load the tokenizer from a pretrained model."""
        return super().from_pretrained(*inputs, **kwargs)

    # add special tokens <bos> and <eos>
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """Build the input with special tokens."""
        bos = [self.bos_token_id]
        eos = [self.eos_token_id]

        if token_ids_1 is None:
            return bos + token_ids_0 + eos
        return bos + token_ids_0 + eos + token_ids_1 + eos
