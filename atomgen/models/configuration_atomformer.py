"""Configuration class for Atomformer."""

from typing import Any

from transformers.configuration_utils import PretrainedConfig


class AtomformerConfig(PretrainedConfig):
    r"""
    Configuration of a :class:`~transform:class:`~transformers.AtomformerModel`.

    It is used to instantiate an Atomformer model according to the specified arguments.
    """

    model_type = "atomformer"

    def __init__(
        self,
        vocab_size: int = 123,
        dim: int = 768,
        num_heads: int = 32,
        depth: int = 12,
        mlp_ratio: int = 1,
        k: int = 128,
        dropout: float = 0.0,
        mask_token_id: int = 0,
        pad_token_id: int = 119,
        bos_token_id: int = 120,
        eos_token_id: int = 121,
        cls_token_id: int = 122,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[no-untyped-call]
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_heads = num_heads
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.k = k

        self.dropout = dropout
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.cls_token_id = cls_token_id
