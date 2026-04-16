import inspect
from typing import Optional, Tuple

from generative.networks.nets import ControlNet, DiffusionModelUNet
import torch
import torch.nn.functional as F
from torch import nn


def _resize(x: torch.Tensor, size: tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
    if x.shape[-2:] == size:
        return x
    if mode in {"bilinear", "bicubic"}:
        return F.interpolate(x, size=size, mode=mode, align_corners=False)
    return F.interpolate(x, size=size, mode=mode)


class FlowMatchingModel(nn.Module):
    def __init__(
        self,
        unet: DiffusionModelUNet,
        controlnet: ControlNet,
        *,
        max_timestep: int = 1000,
        latent_size: Tuple[int, int] = (32, 32),
        output_size: Tuple[int, int] = (10, 10),
    ) -> None:
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        self.max_timestep = int(max_timestep)
        self.latent_size = tuple(latent_size)
        self.output_size = tuple(output_size)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        elif t.ndim == 1 and t.shape[0] == 1 and x.shape[0] != 1:
            t = t.expand(x.shape[0])
        elif t.ndim != 1 or t.shape[0] != x.shape[0]:
            raise ValueError(f"Expected t with shape [{x.shape[0]}], got {tuple(t.shape)}.")

        timesteps = (t * (self.max_timestep - 1)).floor().long()

        x_latent = _resize(x, self.latent_size, mode="bilinear")
        if masks.shape[0] != x.shape[0]:
            raise ValueError(f"Batch mismatch: x {x.shape}, masks {masks.shape}")

        masks_latent = _resize(masks, self.latent_size, mode="nearest")
        down_res, mid_res = self.controlnet(
            x=x_latent,
            timesteps=timesteps,
            controlnet_cond=masks_latent,
        )
        pred = self.unet(
            x=x_latent,
            timesteps=timesteps,
            down_block_additional_residuals=down_res,
            mid_block_additional_residual=mid_res,
        )

        return _resize(pred, self.output_size, mode="bilinear")


def _filter_kwargs(src: dict, ctor) -> dict:
    try:
        params = set(inspect.signature(ctor).parameters)
    except (TypeError, ValueError):
        try:
            params = set(inspect.signature(ctor.__init__).parameters)
        except Exception:
            return {}
    params.discard("self")
    params.discard("args")
    params.discard("kwargs")
    return {k: v for k, v in src.items() if k in params}


def build_model(
    config: Optional[dict] = None,
    device: Optional[torch.device] = None,
    **overrides,
) -> FlowMatchingModel:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = (config or {}).get("model_config", {}).copy()
    model_cfg.update(overrides or {})

    if device.type != "cuda" and model_cfg.get("use_flash_attention", False):
        print("WARNING: disabling use_flash_attention because the model is not on CUDA.")
        model_cfg["use_flash_attention"] = False

    max_timestep = int(model_cfg.get("max_timestep", 1000))
    latent_size = tuple(model_cfg.get("latent_size", (32, 32)))
    output_size = tuple(model_cfg.get("output_size", (10, 10)))
    unet_kwargs = _filter_kwargs(model_cfg, DiffusionModelUNet)
    controlnet_kwargs = _filter_kwargs(model_cfg, ControlNet)
    seeded = {k: v for k, v in unet_kwargs.items() if k not in controlnet_kwargs}
    controlnet_kwargs = {**seeded, **controlnet_kwargs}

    unet = DiffusionModelUNet(**unet_kwargs)

    controlnet_kwargs.pop("out_channels", None)
    controlnet_kwargs.pop("dropout_cattn", None)
    controlnet = ControlNet(**controlnet_kwargs)
    controlnet.load_state_dict(unet.state_dict(), strict=False)

    model = FlowMatchingModel(
        unet=unet,
        controlnet=controlnet,
        max_timestep=max_timestep,
        latent_size=latent_size,
        output_size=output_size,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {trainable} trainable parameters.")
    return model
