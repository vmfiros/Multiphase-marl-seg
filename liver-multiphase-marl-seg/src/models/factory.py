
from .swin_unet_cpaf import SwinUNetCPAF
from .baselines.unet3d import UNet3D
from .baselines.resunet3d import ResUNet3D
from .baselines.swinunet3d import SwinUNet3D

def build_model(arch: str, cfg: dict):
    arch = arch.lower()
    if arch == 'swin_unet_cpaf':
        ch = tuple(cfg['model'].get('channels', [96,192,384]))
        return SwinUNetCPAF(in_channels=2, out_channels=1, channels=ch,
                            use_decoder_swin=cfg['model'].get('use_decoder_swin', True))
    if arch == 'unet3d':
        base = int(cfg['model'].get('base_channels', 32))
        depth = int(cfg['model'].get('depth', 4))
        return UNet3D(in_channels=2, out_channels=1, base_ch=base, depth=depth)
    if arch == 'resunet3d':
        base = int(cfg['model'].get('base_channels', 32))
        depth = int(cfg['model'].get('depth', 4))
        return ResUNet3D(in_channels=2, out_channels=1, base_ch=base, depth=depth)
    if arch == 'swinunet3d':
        ch = tuple(cfg['model'].get('channels', [96,192,384]))
        return SwinUNet3D(in_channels=2, out_channels=1, channels=ch,
                          use_decoder_swin=cfg['model'].get('use_decoder_swin', True))
    raise ValueError(f'Unknown arch: {arch}')
