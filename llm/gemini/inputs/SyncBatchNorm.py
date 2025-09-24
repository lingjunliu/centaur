
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import torch.distributed as dist

def sync_batchnorm_inputs():
    list_of_inputs = []

    input_dict = {
        "num_features": 100,
        "eps": 1e-05,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "process_group": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = sync_batchnorm_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('SyncBatchNorm', generated_inputs)
