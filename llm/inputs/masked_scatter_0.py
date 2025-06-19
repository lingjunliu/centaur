
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def masked_scatter_inputs():
    list_of_inputs = []

    # Input 1: Basic example with float tensors
    input_tensor = torch.randn(2, 3).numpy()
    mask_tensor = (torch.randn(2, 3) > 0).numpy()
    source_tensor = torch.randn(10).numpy()
    input_dict = {"input": input_tensor, "mask": mask_tensor, "source": source_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer tensors
    input_tensor = torch.randint(0, 10, (3, 4)).numpy()
    mask_tensor = (torch.randn(3, 4) > 0).numpy()
    source_tensor = torch.randint(0, 10, (15,)).numpy()
    input_dict = {"input": input_tensor, "mask": mask_tensor, "source": source_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor
    input_tensor = torch.randn(2, 3, 4).numpy()
    mask_tensor = (torch.randn(2, 3, 4) > 0).numpy()
    source_tensor = torch.randn(30).numpy()
    input_dict = {"input": input_tensor, "mask": mask_tensor, "source": source_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Boolean tensors
    input_tensor = (torch.randn(2, 2) > 0).numpy()
    mask_tensor = (torch.randn(2, 2) > 0.5).numpy()
    source_tensor = (torch.randn(5) > 0).numpy()
    input_dict = {"input": input_tensor, "mask": mask_tensor, "source": source_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative values and different mask density
    input_tensor = torch.randn(3, 3).numpy()
    mask_tensor = (torch.randn(3, 3) > 0.8).numpy()
    source_tensor = torch.randn(5).numpy()
    input_dict = {"input": input_tensor, "mask": mask_tensor, "source": source_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

if __name__ == '__main__':
    generated_inputs = {}
    generated_inputs["torch.masked_scatter"] = masked_scatter_inputs()
    print(generated_inputs)

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.masked_scatter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.masked_scatter'.")

check_valid('torch.masked_scatter', generated_inputs['torch.masked_scatter'], lib="torch")
