from pathlib import Path

from huggingface_hub import hf_hub_download
import numpy as np
from pytorch_retrieve import load_model
import torch
import xarray as xr


def download_model():
    """
    Download GPROF-NN 3D model from hugging face.
    """
    repo_id = "simonpf/gprof_nn"
    filename = "gprof_nn_3d.pt"
    file_path = hf_hub_download(repo_id=repo_id, filename=filename)
    return file_path



class Retrieval():
    """
    Interface for running the GPROF-V8 retrieval on the IPWGML SPR dataset.
    """
    def __init__(self, device="cpu", dtype="float32"):
        self.model = load_model(download_model()).eval()
        self.dtype = getattr(torch, dtype)
        self.device = torch.device(device)

    def __call__(self, input_data: xr.Dataset) -> xr.Dataset:
        """
        Run GPROF V8 retrieval on IPWGML input data.
        """
        input_data = input_data.transpose("batch", "channels_gmi", "scan", "pixel")

        tbs = input_data.obs_gmi.data
        eia = input_data.eia_gmi.data
        n_batch, _, n_scans, n_pixels = tbs.shape

        tbs_full = np.zeros((n_batch, 15, n_scans, n_pixels))
        gmi_inds = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14]
        tbs_full[:, gmi_inds] = tbs

        eia_full = np.zeros((n_batch, 15, n_scans, n_pixels))
        eia_full[:, gmi_inds] = eia
        eia_full[:, 5] = eia_full[:, 4]

        anc = np.nan * np.zeros((n_batch, 14, n_scans, n_pixels))

        inpt = {
            "brightness_temperatures": torch.tensor(tbs_full.astype(np.float32)),
            "earth_incidence_angles": torch.tensor(eia_full.astype(np.float32)),
            "ancillary_data": torch.tensor(anc.astype(np.float32)),
        }

        model = self.model.to(dtype=self.dtype, device=self.device)

        with torch.no_grad():
            inpt = {name: tnsr.to(device=self.device, dtype=self.dtype) for name, tnsr in inpt.items()}
            pred = self.model(inpt)

            surface_precip = pred["surface_precip"].expected_value().cpu().numpy()
            pop = pred["surface_precip"].probability_greater_than(1e-3).cpu().numpy()
            precip_flag = pop > 0.5
            pop_heavy = pred["surface_precip"].probability_greater_than(10.0).cpu().numpy()
            heavy_precip_flag = pop_heavy > 0.5

        results = xr.Dataset({
            "surface_precip": (("batch", "scan", "pixel"), surface_precip[:, 0]),
            "probability_of_precip": (("batch", "scan", "pixel"), pop[:, 0]),
            "precip_flag": (("batch", "scan", "pixel"), precip_flag[:, 0]),
            "probability_of_heavy_precip": (("batch", "scan", "pixel"), pop_heavy[:, 0]),
            "heavy_precip_flage": (("batch", "scan", "pixel"), heavy_precip_flag[:, 0]),
        })
        return results
