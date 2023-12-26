from typing import Optional, Union
import torch
from torch import nn
from diffusers.models.modeling_utils import get_parameter_device, get_parameter_dtype


class AbstractSCTunerLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.dim = dim


class SCTunerLinearLayer(AbstractSCTunerLayer):
    r"""
    A linear layer that is used with SCEdit.

    Parameters:
        dim (`int`):
            Number of dim.
        out_features (`int`):
            Number of output features.
        rank (`int`, `optional`, defaults to 4):
            The rank of the LoRA layer.
        device (`torch.device`, `optional`, defaults to `None`):
            The device to use for the layer's weights.
        dtype (`torch.dtype`, `optional`, defaults to `None`):
            The dtype to use for the layer's weights.
    """

    def __init__(
        self,
        dim: int,
        rank: Optional[int] = None,
        scale: float = 1.0,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(dim=dim)
        if rank is None:
            rank = dim
        self.down = nn.Linear(dim, rank, device=device, dtype=dtype)
        self.up = nn.Linear(rank, dim, device=device, dtype=dtype)
        self.act = nn.GELU()
        self.rank = rank
        self.scale = scale

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        hidden_states_input = hidden_states.permute(0, 2, 3, 1)
        down_hidden_states = self.down(hidden_states_input.to(dtype))
        up_hidden_states = self.up(self.act(down_hidden_states))
        up_hidden_states = up_hidden_states.to(orig_dtype).permute(0, 3, 1, 2)
        return self.scale * up_hidden_states + hidden_states


class SCTunerLinearLayer2(AbstractSCTunerLayer):
    def __init__(
        self,
        dim: int,
        rank: Optional[int] = None,
        scale: float = 1.0,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(dim=dim)
        if rank is None:
            rank = dim
        self.model = nn.Sequential(
            nn.Linear(dim, rank, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(rank, rank, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(rank, dim, device=device, dtype=dtype),
        )
        self.rank = rank
        self.scale = scale

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.model[0].weight.dtype

        hidden_states_input = hidden_states.permute(0, 2, 3, 1)
        hidden_states_output = self.model(hidden_states_input.to(dtype))
        hidden_states_output = hidden_states_output.to(orig_dtype).permute(0, 3, 1, 2)
        return self.scale * hidden_states_output + hidden_states


class SCTuner(nn.Module):
    def __init__(self):
        super().__init__()
        self.sc_tuners = None
        self.res_skip_channels = None

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        return get_parameter_device(self)

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

    def set_sctuner(self, sctuner_module=SCTunerLinearLayer, **kwargs):
        assert isinstance(self.res_skip_channels, list)
        sc_tuners = []
        for c in self.res_skip_channels:
            sc_tuners.append(
                sctuner_module(dim=c, device=self.device, dtype=self.dtype, **kwargs)
            )
        self.sc_tuners = nn.ModuleList(sc_tuners)
