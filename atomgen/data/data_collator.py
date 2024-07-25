"""Data collator for atom modeling."""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn.functional as f
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer
from transformers.data.data_collator import DataCollatorMixin, _torch_collate_batch


@dataclass
class DataCollatorForAtomModeling(DataCollatorMixin):  # type: ignore
    """
    Data collator used for atom modeling tasks in molecular representations.

    This collator prepares input data for various atom modeling tasks, including
    masked atom modeling (MAM), autoregressive modeling, and coordinate perturbation.
    It supports both padding and flattening of input data.

    Args:
        tokenizer (PreTrainedTokenizer): Tokenizer used for encoding the data.
        mam (Union[bool, float]): If True, uses original masked atom modeling.
                                  If float, masks a constant fraction of atoms/tokens.
        autoregressive (bool): Whether to use autoregressive modeling.
        coords_perturb (float): Standard deviation for coordinate perturbation.
        return_lap_pe (bool): Whether to return Laplacian positional encoding.
        return_edge_indices (bool): Whether to return edge indices.
        k (int): Number of eigenvectors to use for Laplacian positional encoding.
        max_radius (float): Maximum distance for edge cutoff.
        max_neighbors (int): Maximum number of neighbors.
        pad (bool): Whether to pad the input data.
        pad_to_multiple_of (Optional[int]): Pad to multiple of this value.
        return_tensors (str): Return tensors as "pt" or "tf".

    Attributes
    ----------
        tokenizer (PreTrainedTokenizer): The tokenizer used for encoding.
        mam (Union[bool, float]): The masked atom modeling setting.
        autoregressive (bool): The autoregressive modeling setting.
        coords_perturb (float): The coordinate perturbation standard deviation.
        return_lap_pe (bool): The Laplacian positional encoding setting.
        return_edge_indices (bool): The edge indices return setting.
        k (int): The number of eigenvectors for Laplacian PE.
        max_radius (float): The maximum distance for edge cutoff.
        max_neighbors (int): The maximum number of neighbors.
        pad (bool): The padding setting.
        pad_to_multiple_of (Optional[int]): The multiple for padding.
        return_tensors (str): The tensor return format.
    """

    tokenizer: PreTrainedTokenizer
    mam: Union[bool, float] = True
    autoregressive: bool = False
    coords_perturb: float = 0.0
    return_lap_pe: bool = False
    return_edge_indices: bool = False
    k: int = 16
    max_radius: float = 12.0
    max_neighbors: int = 20
    pad: bool = True
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    # ruff: noqa: PLR0912
    def torch_call(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Collate a batch of samples.

        Args:
            examples: List of samples to collate.

        Returns
        -------
            Dict[str, Any]: Dictionary of batched data.

        """
        # Handle dict or lists with proper padding and conversion to tensor.
        if self.pad:
            if isinstance(examples[0], Mapping):
                batch: Dict[str, Any] = self.tokenizer.pad(
                    examples,
                    return_tensors="pt",
                    pad_to_multiple_of=self.pad_to_multiple_of,
                )
            else:
                batch = {
                    "input_ids": _torch_collate_batch(
                        examples,
                        self.tokenizer,
                        pad_to_multiple_of=self.pad_to_multiple_of,
                    )
                }

            if self.return_lap_pe:
                # Compute Laplacian and positional encoding
                (
                    batch["node_pe"],
                    batch["edge_pe"],
                    batch["attention_mask"],
                ) = self.torch_compute_lap_pe(batch["coords"], batch["attention_mask"])
            if self.return_edge_indices:
                # Compute edge indices and distances
                (
                    batch["edge_indices"],
                    batch["edge_distances"],
                    batch["attention_mask"],
                ) = self.torch_compute_edges(batch["coords"], batch["attention_mask"])
        else:
            # flatten all lists in examples and concatenate
            batch = self.flatten_batch(examples)

        t = torch.zeros(batch["input_ids"].shape[0]).float().uniform_(0, 1)
        t = torch.cos(t * math.pi * 0.5)

        if self.mam:
            special_tokens_mask = batch.pop("special_tokens_mask", None)
            if special_tokens_mask is None:
                special_tokens_mask = [
                    self.tokenizer.get_special_tokens_mask(
                        val, already_has_special_tokens=True
                    )
                    for val in batch["input_ids"].tolist()
                ]
                special_tokens_mask = torch.tensor(
                    special_tokens_mask, dtype=torch.bool
                )
            else:
                special_tokens_mask = special_tokens_mask.bool()

            if isinstance(self.mam, float):
                # Constant masking of a float fraction of the atoms/tokens
                mask = torch.bernoulli(
                    torch.full(batch["input_ids"].shape, self.mam)
                ).bool()
                batch["input_ids"], batch["labels"] = self.apply_mask(
                    batch["input_ids"], mask, special_tokens_mask
                )
            else:
                # Original MaskGIT functionality
                batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                    batch["input_ids"], t, special_tokens_mask=special_tokens_mask
                )

        if self.autoregressive:
            # extend coords
            batch["coords"] = torch.cat(
                [
                    torch.zeros_like(batch["coords"][:, :1]),
                    batch["coords"],
                    torch.zeros_like(batch["coords"][:, :1]),
                ],
                dim=1,
            )
            if "labels" not in batch:
                batch["labels"] = batch["input_ids"].clone()
                batch["labels_coords"] = batch["coords"].clone()

            # create mask of ~special_tokens_mask and exclude bos and eos tokens
            special_tokens_mask[batch["labels"] == self.tokenizer.bos_token_id] = False
            special_tokens_mask[batch["labels"] == self.tokenizer.eos_token_id] = False
            batch["labels"] = torch.where(~special_tokens_mask, batch["labels"], -100)

        if self.coords_perturb > 0:
            batch["coords"], batch["labels_coords"] = self.torch_perturb_coords(
                batch["coords"],
                batch.get("fixed", None),
                self.coords_perturb,
            )

        return batch

    def torch_mask_tokens(
        self, inputs: Any, t: Any, special_tokens_mask: Optional[Any] = None
    ) -> Tuple[Any, Any]:
        """Prepare masked tokens inputs/labels for masked atom modeling."""
        labels = inputs.clone()

        batch, seq_len = inputs.shape
        num_token_masked = (seq_len * t).round().clamp(min=1)
        batch_randperm = torch.rand((batch, seq_len)).argsort(dim=-1)
        mask = batch_randperm < num_token_masked.unsqueeze(1)
        inputs = torch.where(
            mask,
            inputs,
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token),
        )
        labels = torch.where(~mask, labels, -100)
        if special_tokens_mask is not None:
            labels = torch.where(~special_tokens_mask, labels, -100)

        return inputs, labels

    def apply_mask(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor,
        special_tokens_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the mask to the input tokens."""
        labels = inputs.clone()
        inputs = torch.where(
            mask,
            torch.tensor(self.tokenizer.mask_token_id, device=inputs.device),
            inputs,
        )
        labels = torch.where(
            ~mask | special_tokens_mask,
            torch.tensor(-100, device=labels.device),
            labels,
        )
        return inputs, labels

    def torch_perturb_coords(
        self, inputs: torch.Tensor, fixed: Optional[torch.Tensor], perturb_std: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare perturbed coords inputs/labels for coordinate denoising."""
        if fixed is None:
            fixed = torch.zeros_like(inputs).bool()
        labels = inputs.clone()
        noise = torch.empty_like(inputs).normal_(0, perturb_std)
        inputs[~fixed.bool()] += noise[~fixed.bool()]
        return inputs, labels

    def flatten_batch(self, examples: Any) -> Dict[str, Any]:
        """Flatten all lists in examples and concatenate with batch indicator."""
        batch = {}
        for key in examples[0]:
            if key == "input_ids":
                lengths = []
                for sample in examples:
                    lengths.append(len(sample[key]))
                batch["batch"] = torch.arange(len(examples)).repeat_interleave(
                    torch.tensor(lengths)
                )
                batch[key] = torch.cat(
                    [torch.tensor(sample[key]) for sample in examples], dim=0
                )
            elif (
                key.startswith("coords")
                or key.endswith("coords")
                or (key.startswith("fixed") or key.endswith("fixed"))
            ):
                batch[key] = torch.cat(
                    [torch.tensor(sample[key]) for sample in examples], dim=0
                )
            elif key.startswith("energy") or key.endswith("energy"):
                batch[key] = torch.tensor([sample[key] for sample in examples])
            elif key.startswith("forces") or key.endswith("forces"):
                batch[key] = torch.cat(
                    [torch.tensor(sample[key]) for sample in examples], dim=0
                )
        return batch

    def torch_compute_edges(self, coords: Any, attention_mask: Any) -> Any:
        """Compute edge indices and distances for each batch."""
        dist_matrix = torch.cdist(coords, coords, p=2)
        b, n, _ = dist_matrix.shape

        # ignore distance in padded coords by setting to large number
        attention_mask_mult = (1.0 - attention_mask) * 1e6
        dist_matrix = dist_matrix + attention_mask_mult.unsqueeze(1)
        dist_matrix = dist_matrix + attention_mask_mult.unsqueeze(2)

        # to avoid self-loop, set diagonal to a large number
        dist_matrix = dist_matrix + torch.eye(n) * 1e6

        # get adjacency matrix using cutoff
        adjacency_matrix = torch.where(dist_matrix <= self.max_radius, 1, 0).float()

        # set max_num_neighbors to 20 to get closest 20 neighbors and set rest to zero
        _, topk_indices = torch.topk(
            dist_matrix,
            k=min(self.max_neighbors, dist_matrix.size(2)),
            dim=2,
            largest=False,
        )
        mask = torch.zeros_like(dist_matrix)
        mask.scatter_(2, topk_indices, 1)
        adjacency_matrix *= mask

        # get distances for each batch in for loop
        distance_list = []
        for bi in range(b):
            distance = dist_matrix[bi][adjacency_matrix[bi] != 0]
            distance_list.append(distance)

        # get edge_indices for each batch in for loop
        edge_indices_list = []
        lengths = []
        for bi in range(b):
            edge_indices = torch.column_stack(torch.where(adjacency_matrix[bi] != 0))
            lengths.append(edge_indices.size(0))
            edge_indices_list.append(edge_indices)

        edge_indices = pad_sequence(
            edge_indices_list, batch_first=True, padding_value=0
        )
        edge_distances = pad_sequence(distance_list, batch_first=True, padding_value=-1)
        edge_attention_mask = torch.cat(
            [
                torch.cat(
                    [
                        torch.ones(1, length),
                        torch.zeros(1, edge_indices.size(1) - length),
                    ],
                    dim=1,
                )
                for length in lengths
            ],
            dim=0,
        )
        attention_mask = torch.cat([attention_mask, edge_attention_mask], dim=1)

        return edge_indices, edge_distances, attention_mask

    def torch_compute_lap_pe(self, coords: Any, attention_mask: Any) -> Any:
        """Compute Laplacian positional encoding for each batch."""
        dist_matrix = torch.cdist(coords, coords, p=2)
        b, n, _ = dist_matrix.shape

        # ignore distance in padded coords by setting to large number
        attention_mask_mult = (1.0 - attention_mask) * 1e6
        dist_matrix = dist_matrix + attention_mask_mult.unsqueeze(1)
        dist_matrix = dist_matrix + attention_mask_mult.unsqueeze(2)

        # to avoid self-loop, set diagonal to a large number
        dist_matrix = dist_matrix + torch.eye(n) * 1e6

        # get adjacency matrix using cutoff
        adjacency_matrix = torch.where(dist_matrix <= self.max_radius, 1, 0).float()

        # set max_num_neighbors to 20 to get closest 20 neighbors and set rest to zero
        _, topk_indices = torch.topk(
            dist_matrix,
            k=min(self.max_neighbors, dist_matrix.size(2)),
            dim=2,
            largest=False,
        )
        mask = torch.zeros_like(dist_matrix)
        mask.scatter_(2, topk_indices, 1)
        adjacency_matrix *= mask

        # get distances for each batch in for loop
        distance_list = []
        for bi in range(b):
            distance = dist_matrix[bi][adjacency_matrix[bi] != 0]
            distance_list.append(distance)

        # get edge_indices for each batch in for loop
        edge_indices_list = []
        for bi in range(b):
            edge_indices = torch.column_stack(torch.where(adjacency_matrix[bi] != 0))
            edge_indices_list.append(edge_indices)

        # Construct graph Laplacian for each batch
        degree_matrix = torch.diag_embed(adjacency_matrix.sum(dim=2).clip(1) ** -0.5)
        laplacian_matrix = (
            torch.eye(n) - degree_matrix @ adjacency_matrix @ degree_matrix
        )

        # Eigenvalue decomposition for each batch
        eigval, eigvec = torch.linalg.eigh(laplacian_matrix)

        eigvec = eigvec.float()  # [N, N (channels)]
        eigval = torch.sort(torch.abs(torch.real(eigval)))[0].float()  # [N (channels),]

        if eigvec.size(1) < self.k:
            node_pe = f.pad(eigvec, (0, self.k - eigvec.size(2), 0, 0))
        else:
            # use smallest eigenvalues
            node_pe = eigvec[:, :, : self.k]

        all_edges_pe_list = []
        lengths = []
        for i, edge_indices in enumerate(edge_indices_list):
            e = edge_indices.shape[0]
            lengths.append(e)
            all_edges_pe = torch.zeros([e, 2 * self.k])
            all_edges_pe[:, : self.k] = torch.index_select(
                node_pe[i], 0, edge_indices[:, 0]
            )
            all_edges_pe[:, self.k :] = torch.index_select(
                node_pe[i], 0, edge_indices[:, 1]
            )
            all_edges_pe_list.append(all_edges_pe)

        # get attention mask for edge_pe based on all_edges_pe_list

        edge_pe = pad_sequence(all_edges_pe_list, batch_first=True, padding_value=0)
        edge_attention_mask = torch.cat(
            [
                torch.cat(
                    [torch.ones(1, length), torch.zeros(1, edge_pe.size(1) - length)],
                    dim=1,
                )
                for length in lengths
            ],
            dim=0,
        )
        attention_mask = torch.cat([attention_mask, edge_attention_mask], dim=1)

        edge_distances = pad_sequence(distance_list, batch_first=True, padding_value=-1)
        edge_pe = torch.cat([edge_pe, edge_distances.unsqueeze(-1)], dim=2)

        node_pe = torch.cat([node_pe, node_pe], dim=2)

        return node_pe, edge_pe, attention_mask
