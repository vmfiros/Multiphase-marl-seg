
from __future__ import annotations
import numpy as np
import SimpleITK as sitk

def resample_to_spacing(img: sitk.Image, spacing=(1.0,1.0,1.0), is_label=False):
    orig_spacing = img.GetSpacing()
    orig_size = img.GetSize()
    new_spacing = spacing
    new_size = [int(round(osz*ospc/nspc)) for osz,ospc,nspc in zip(orig_size, orig_spacing, new_spacing)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(img)

def clip_and_normalize_hu(arr: np.ndarray, hu_min=-200, hu_max=300, mode="zscore"):
    arr = np.clip(arr, hu_min, hu_max).astype(np.float32)
    if mode == "zscore":
        mu = arr.mean()
        sigma = arr.std() + 1e-8
        arr = (arr - mu)/sigma
    elif mode == "minmax":
        arr = (arr - hu_min)/(hu_max - hu_min)
    return arr

def sitk_to_np(img: sitk.Image):
    return sitk.GetArrayFromImage(img)  # z,y,x order

def np_to_sitk(arr: np.ndarray, ref: sitk.Image):
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(ref)
    return out
