
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def masked_scatter_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor, boolean mask, float source
    input_tensor = torch.randn(3, 4).numpy()
    mask_tensor = (torch.rand(3, 4) > 0.5).numpy()
    source_tensor = torch.randn(mask_tensor.sum()).numpy()
    input_dict = {"input": input_tensor, "mask": mask_tensor, "source": source_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Int tensor, boolean mask, int source
    input_tensor = torch.randint(-5, 5, (2, 2)).numpy()
    mask_tensor = (torch.rand(2, 2) > 0.5).numpy()
    source_tensor = torch.randint(-5, 5, (mask_tensor.sum(),)).numpy()
    input_dict = {"input": input_tensor, "mask": mask_tensor, "source": source_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor, boolean mask, float source
    input_tensor = torch.randn(2, 3, 2).numpy()
    mask_tensor = (torch.rand(2, 3, 2) > 0.5).numpy()
    source_tensor = torch.randn(mask_tensor.sum()).numpy()
    input_dict = {"input": input_tensor, "mask": mask_tensor, "source": source_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Negative values in input and source
    input_tensor = torch.randn(2, 2) * -1.0
    input_tensor = input_tensor.numpy()
    mask_tensor = (torch.rand(2, 2) > 0.5).numpy()
    source_tensor = (torch.randn(mask_tensor.sum()) * -1.0).numpy()
    input_dict = {"input": input_tensor, "mask": mask_tensor, "source": source_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 1D tensors
    input_tensor = torch.randn(5).numpy()
    mask_tensor = (torch.rand(5) > 0.5).numpy()
    source_tensor = torch.randn(mask_tensor.sum()).numpy()
    input_dict = {"input": input_tensor, "mask": mask_tensor, "source": source_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Complex tensor
    input_tensor = torch.randn(2, 2, dtype=torch.complex64).numpy()
    mask_tensor = (torch.rand(2, 2) > 0.5).numpy()
    source_tensor = torch.randn(mask_tensor.sum(), dtype=torch.complex64).numpy()
    input_dict = {"input": input_tensor, "mask": mask_tensor, "source": source_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = masked_scatter_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('masked_scatter', generated_inputs)
