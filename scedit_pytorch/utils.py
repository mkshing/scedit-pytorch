import os
import logging
from typing import Dict, Union

import torch
import safetensors
from diffusers.utils import DIFFUSERS_CACHE, HF_HUB_OFFLINE, _get_model_file
from . import UNet2DConditionModel

logger = logging.getLogger(__name__)

SCEDIT_WEIGHT_NAME_SAFE = "pytorch_scedit_weights.safetensors"


def save_function(weights, filename):
    return safetensors.torch.save_file(weights, filename, metadata={"format": "pt"})


def save_scedit(state_dict: Dict[str, torch.Tensor], save_directory: str):
    if os.path.isfile(save_directory):
        logger.error(
            f"Provided path ({save_directory}) should be a directory, not a file"
        )
        return
    os.makedirs(save_directory, exist_ok=True)
    save_function(state_dict, os.path.join(save_directory, SCEDIT_WEIGHT_NAME_SAFE))
    logger.info(
        f"Model weights saved in {os.path.join(save_directory, SCEDIT_WEIGHT_NAME_SAFE)}"
    )


def scedit_state_dict(
    pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]], **kwargs
) -> Dict[str, torch.Tensor]:
    cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
    force_download = kwargs.pop("force_download", False)
    resume_download = kwargs.pop("resume_download", False)
    proxies = kwargs.pop("proxies", None)
    local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
    use_auth_token = kwargs.pop("use_auth_token", None)
    revision = kwargs.pop("revision", None)
    subfolder = kwargs.pop("subfolder", None)
    weight_name = kwargs.pop("weight_name", None)
    user_agent = {
        "file_type": "attn_procs_weights",
        "framework": "pytorch",
    }

    model_file = None
    if not isinstance(pretrained_model_name_or_path_or_dict, dict):
        # Here we're relaxing the loading check to enable more Inference API
        # friendliness where sometimes, it's not at all possible to automatically
        # determine `weight_name`.
        if weight_name is None:
            weight_name = SCEDIT_WEIGHT_NAME_SAFE
        model_file = _get_model_file(
            pretrained_model_name_or_path_or_dict,
            weights_name=weight_name,
            cache_dir=cache_dir,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            use_auth_token=use_auth_token,
            revision=revision,
            subfolder=subfolder,
            user_agent=user_agent,
        )
        state_dict = safetensors.torch.load_file(model_file, device="cpu")
    else:
        state_dict = pretrained_model_name_or_path_or_dict

    return state_dict


def load_scedit_into_unet(
    state_dict: Union[str, Dict[str, torch.Tensor]],
    unet: UNet2DConditionModel,
    **kwargs,
) -> UNet2DConditionModel:
    if isinstance(state_dict, str):
        state_dict = scedit_state_dict(state_dict, **kwargs)
    assert unet.has_sctuner, "Make sure to call `set_sctuner` before!"
    unet.load_state_dict(state_dict, strict=False)
    return unet
