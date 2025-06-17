
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy

def set_module_inputs():
    list_of_inputs = []

    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)

    module1 = MyModule()
    
    class AnotherModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, kernel_size=3)

        def forward(self, x):
            return self.conv(x)

    module2 = AnotherModule()

    input1 = {
        "mod": "my_module",
        "new_module": module1
    }
    list_of_inputs.append(copy.deepcopy(input1))


    input2 = {
        "mod": "conv_module",
        "new_module": module2
    }
    list_of_inputs.append(copy.deepcopy(input2))


    return list_of_inputs

generated_inputs["torch.jit.set_module"] = set_module_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.jit.set_module' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.jit.set_module'.")

check_valid('torch.jit.set_module', generated_inputs['torch.jit.set_module'], lib="torch")
