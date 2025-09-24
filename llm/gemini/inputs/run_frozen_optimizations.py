
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

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

generated_inputs = run_frozen_optimizations_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('run_frozen_optimizations', generated_inputs)
