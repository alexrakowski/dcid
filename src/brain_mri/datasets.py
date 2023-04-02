import numpy as np
import torch.utils.data as torch_data
import nibabel as nib
from scipy.ndimage import zoom


def zoom_scan(scan, new_size):
    factors = (
        new_size / scan.shape[0],
        new_size / scan.shape[1],
        new_size / scan.shape[2]
    )

    return zoom(
        scan,
        factors,
        # prefilter causes weird artifacts in empty slices of the scans
        prefilter=False,
    )


class BrainMRIDataset(torch_data.Dataset):
    def __init__(
            self,
            df,
            y1_col_name,
            y2_col_name,
            img_size,
            slice_dim,
    ):
        self.df = df
        self.img_size = img_size
        self.y1_col_name = y1_col_name
        self.y2_col_name = y2_col_name
        self.slice_dim = slice_dim

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        idx = int(idx)
        row = self.df.iloc[idx]

        scan = nib.load(row['path']).get_fdata()
        if self.img_size is not None:
            scan = zoom_scan(scan, self.img_size)
        if self.slice_dim is not None:
            scan = np.take(scan, scan.shape[self.slice_dim] // 2, self.slice_dim)
        scan = (scan - scan.min()) / scan.max()

        scan = np.expand_dims(scan, 0).astype(np.float32)

        y = row[[self.y1_col_name, self.y2_col_name]].values.astype(np.float32)

        return scan, y

    @property
    def num_channels(self):
        return 1
