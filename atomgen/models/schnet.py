from torch import nn
from torch_geometric.nn import SchNet
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel


class SchNetConfig(PretrainedConfig):
    model_type = "transformer"

    def __init__(
        self,
        vocab_size=122,
        hidden_channels=128,
        num_filters=128,
        num_interactions=6,
        num_gaussians=50,
        cutoff=10.0,
        interaction_graph=None,
        max_num_neighbors=32,
        readout="add",
        dipole=False,
        mean=None,
        std=None,
        atomref=None,
        mask_token_id=119,
        pad_token_id=120,
        bos_token_id=121,
        eos_token_id=122,
        **kwargs,
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


class SchNetPreTrainedModel(PreTrainedModel):
    config_class = SchNetConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False


class SchNetModel(SchNetPreTrainedModel):
    def __init__(self, config):
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
        input_ids,
        coords,
        batch,
        labels_energy=None,
        fixed=None,
        attention_mask=None,
    ):
        energy_pred = self.model(z=input_ids, pos=coords, batch=batch)

        loss = None
        if labels_energy is not None:
            labels_energy = labels_energy.to(energy_pred.device)
            loss_fct = nn.L1Loss()
            loss = loss_fct(energy_pred.squeeze(-1), labels_energy)
        return loss, energy_pred
