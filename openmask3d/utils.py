import os
import warnings
from tempfile import NamedTemporaryFile
import numpy as np
import torch
import glob
from datetime import date

def get_free_gpu(min_mem=20000):
    torch.cuda.empty_cache()
    try:
        with NamedTemporaryFile() as f:
            os.system(f"nvidia-smi -q -d Memory | grep -A5 GPU | grep Free > {f.name}")
            memory_available = [int(x.split()[2]) for x in open(f.name, 'r').readlines()]
        if max(memory_available) < min_mem:
            warnings.warn("Not enough memory on GPU, using CPU")
            return torch.device("cpu")
        return torch.device("cuda", np.argmax(memory_available))
    except:
        warnings.warn("Could not get free GPU, using CPU")
        return torch.device("cpu")
    
def create_out_folder(experiment_name: str, 
                      output_path: str = "outputs"):
    date_str = date.today().strftime("%Y-%m-%d-%H:%M:%S")
    folder_name = date_str + '-' + experiment_name
    out_folder = os.path.join(output_path, folder_name)    
    os.makedirs(out_folder, exist_ok=True)
    return out_folder