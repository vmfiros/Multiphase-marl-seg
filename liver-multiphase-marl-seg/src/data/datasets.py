
from __future__ import annotations
import pathlib, numpy as np, nibabel as nib, torch
from torch.utils.data import Dataset

class MultiPhaseCTDataset(Dataset):
    """Assumes folder layout:
    root_interim/patient_id/{arterial.nii.gz, venous.nii.gz, label.nii.gz}
    """
    def __init__(self, root_interim, ids, transform=None):
        self.root = pathlib.Path(root_interim)
        self.ids = list(ids)
        self.transform = transform

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        pdir = self.root / pid
        a = nib.load(str(pdir/'arterial.nii.gz')).get_fdata().astype(np.float32)
        v = nib.load(str(pdir/'venous.nii.gz')).get_fdata().astype(np.float32)
        x = np.stack([a, v], axis=0)  # (2, Z, Y, X) nib is ZYX
        y = nib.load(str(pdir/'label.nii.gz')).get_fdata().astype(np.uint8)
        y = np.expand_dims(y, 0)
        sample = {'image': x, 'label': y, 'id': pid}
        if self.transform: sample = self.transform(sample)
        # convert to torch
        img = torch.from_numpy(sample['image'].copy())
        lbl = torch.from_numpy(sample['label'].copy())
        return img, lbl, pid
