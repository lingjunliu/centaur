
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy

def run_frozen_optimizations_inputs():
    list_of_inputs = []

    # Example 1: Simple linear model
    class LinearModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)

    model = LinearModel()
    scripted_module = torch.jit.script(model)
    input_dict = {"mod": scripted_module}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.jit.run_frozen_optimizations"] = run_frozen_optimizations_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.jit.run_frozen_optimizations' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.jit.run_frozen_optimizations'.")

check_valid('torch.jit.run_frozen_optimizations', generated_inputs['torch.jit.run_frozen_optimizations'], lib="torch")
