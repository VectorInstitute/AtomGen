"""SchNet model for energy prediction."""

from typing import Any, Callable, Optional, Tuple

import torch
from torch import nn
from torch_geometric.nn import SchNet
from torch_geometric.typing import OptTensor
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel


class SchNetConfig(PretrainedConfig):  # type: ignore
    r"""
    Stores the configuration of a :class:`~transformers.SchNetModel`.

    It is used to instantiate an SchNet model according to the specified arguments,
    defining the model architecture.

    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 122):
            The size of the vocabulary, used to define the size
            of the output embeddings.

        hidden_channels (:obj:`int`, `optional`, defaults to 128):
            The hidden size of the model.

    model_type = "transformer"

    Attributes
    ----------
        vocab_size (:obj:`int`):
            The size of the vocabulary, used to define
            the size of the output embeddings.

        hidden_channels (:obj:`int`):
            The hidden size of the model.

        num_filters (:obj:`int`):
            The number of filters.

        num_interactions (:obj:`int`):
            The number of interactions.

        num_gaussians (:obj:`int`):
            The number of gaussians.

        cutoff (:obj:`float`):
            The cutoff value.

        interaction_graph (:obj:`str`, `optional`):
            The interaction graph.

        max_num_neighbors (:obj:`int`):
            The maximum number of neighbors.

        readout (:obj:`str`, `optional`):
            The readout method.

        dipole (:obj:`bool`, `optional`):
            Whether to include dipole.

        mean (:obj:`float`, `optional`):
            The mean value.

        std (:obj:`float`, `optional`):
            The standard deviation value.

        atomref (:obj:`float`, `optional`):
            The atom reference value.

        mask_token_id (:obj:`int`, `optional`):
            The token ID for masking.

        pad_token_id (:obj:`int`, `optional`):
            The token ID for padding.

        bos_token_id (:obj:`int`, `optional`):
            The token ID for the beginning of sequence.

        eos_token_id (:obj:`int`, `optional`):
            The token ID for the end of sequence.

    """

    def __init__(
        self,
        vocab_size: int = 123,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        interaction_graph: Optional[Callable[..., Any]] = None,
        max_num_neighbors: int = 32,
        readout: str = "add",
        dipole: bool = False,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        atomref: Optional[OptTensor] = None,
        mask_token_id: int = 0,
        pad_token_id: int = 119,
        bos_token_id: int = 120,
        eos_token_id: int = 121,
        cls_token_id: int = 122,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.interaction_graph = interaction_graph
        self.max_num_neighbors = max_num_neighbors
        self.readout = readout
        self.dipole = dipole
        self.mean = mean
        self.std = std
        self.atomref = atomref
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.cls_token_id = cls_token_id


class SchNetPreTrainedModel(PreTrainedModel):  # type: ignore
    """
    A base class for all SchNet models.

    An abstract class to handle weights initialization and a
    simple interface for loading and exporting models.
    """

    config_class = SchNetConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False


class SchNetModel(SchNetPreTrainedModel):
    """
    SchNet model for energy prediction.

    Args:
        config (:class:`~transformers.SchNetConfig`):
            Configuration class to store the configuration of a model.
    """

    def __init__(self, config: SchNetConfig):
        super().__init__(config)
        self.config = config
        self.model = SchNet(
            hidden_channels=config.hidden_channels,
            num_filters=config.num_filters,
            num_interactions=config.num_interactions,
            num_gaussians=config.num_gaussians,
            cutoff=config.cutoff,
            interaction_graph=config.interaction_graph,
            max_num_neighbors=config.max_num_neighbors,
            readout=config.readout,
            dipole=config.dipole,
            mean=config.mean,
            std=config.std,
            atomref=config.atomref,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        coords: torch.Tensor,
        batch: torch.Tensor,
        labels_energy: Optional[torch.Tensor] = None,
        fixed: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass of the SchNet model.

        Args:
            input_ids (:obj:`torch.Tensor` of shape :obj:`(batch_size, num_atoms)`):
                The input tensor containing the atom indices.

            coords (:obj:`torch.Tensor` of shape :obj:`(num_atoms, 3)`):
                The input tensor containing the atom coordinates.

            batch (:obj:`torch.Tensor` of shape :obj:`(num_atoms)`):
                The input tensor containing the batch indices.

            labels_energy (:obj:`torch.Tensor`, `optional`):
                The input tensor containing the energy labels.

            fixed (:obj:`torch.Tensor`, `optional`):
                The input tensor containing the fixed mask.

            attention_mask (:obj:`torch.Tensor`, `optional`):
                The attention mask for the transformer.

        Returns
        -------
            :obj:`tuple`:
                A tuple of the loss and the energy prediction.
        """
        energy_pred: torch.Tensor = self.model(z=input_ids, pos=coords, batch=batch)

        loss = None
        if labels_energy is not None:
            labels_energy = labels_energy.to(energy_pred.device)
            loss_fct = nn.L1Loss()
            loss = loss_fct(energy_pred.squeeze(-1), labels_energy)
        return loss, energy_pred
