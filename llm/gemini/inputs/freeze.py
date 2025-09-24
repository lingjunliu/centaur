
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import torch.nn as nn
import torch.jit
import copy

def freeze_inputs():
    list_of_inputs = []

    class MyModule1(nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(out_features, in_features))
            self.linear = nn.Linear(out_features, out_features)

        def forward(self, input):
            output = self.weight.mm(input)
            output = self.linear(output)
            return output

    scripted_module1 = torch.jit.script(MyModule1(2, 3).eval())
    input_dict1 = {
        "mod": scripted_module1,
        "preserved_attrs": None,
        "optimize_numerics": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    class MyModule2(nn.Module):
        def __init__(self):
            super().__init__()
            self.modified_tensor = torch.tensor(10)
            self.version = 1

        def forward(self, input):
            self.modified_tensor += 1
            return input + self.modified_tensor

    scripted_module2 = torch.jit.script(MyModule2().eval())
    input_dict2 = {
        "mod": scripted_module2,
        "preserved_attrs": ["version"],
        "optimize_numerics": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    class MyModule3(nn.Module):
        def __init__(self):
            super().__init__()
            self.submodule = nn.Linear(5, 5)
            self.submodule.version = 1

        def forward(self, input):
            return self.submodule(input)

    scripted_module3 = torch.jit.script(MyModule3().eval())
    input_dict3 = {
        "mod": scripted_module3,
        "preserved_attrs": ["submodule.version"],
        "optimize_numerics": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    class MyModule4(nn.Module):
        def __init__(self):
            super().__init__()
            self.my_list = [1, 2, 3]

        def forward(self, input):
            return input

    scripted_module4 = torch.jit.script(MyModule4().eval())
    input_dict4 = {
        "mod": scripted_module4,
        "preserved_attrs": ["my_list"],
        "optimize_numerics": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    class MyModule5(nn.Module):
        def __init__(self):
            super().__init__()
            self.const_tensor = torch.randn(3,3)

        def forward(self, input):
            return self.const_tensor @ input

    scripted_module5 = torch.jit.script(MyModule5().eval())
    input_dict5 = {
        "mod": scripted_module5,
        "preserved_attrs": [],
        "optimize_numerics": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = freeze_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('freeze', generated_inputs)
