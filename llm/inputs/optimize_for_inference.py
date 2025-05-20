
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy

def optimize_for_inference_inputs():
    list_of_inputs = []

    # Example 1: Simple float tensor
    mod = torch.jit.script(torch.nn.Linear(10, 5))
    mod = torch.jit.freeze(mod.eval())
    list_of_inputs.append({"mod": mod})

    # Example 2: Int tensor
    class IntModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x):
            return x + 1
    mod = torch.jit.script(IntModel())
    mod = torch.jit.freeze(mod.eval())
    list_of_inputs.append({"mod": mod})

    # Example 3: More complex model
    class ComplexModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(5, 10)
            self.linear2 = torch.nn.Linear(10, 5)

        def forward(self, x):
            x = torch.relu(self.linear1(x))
            x = torch.sigmoid(self.linear2(x))
            return x

    mod = torch.jit.script(ComplexModel())
    mod = torch.jit.freeze(mod.eval())
    list_of_inputs.append({"mod": mod})

    # Example 4: Module with dropout
    class DropoutModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 5)
            self.dropout = torch.nn.Dropout(p=0.5)

        def forward(self, x):
            x = self.linear(x)
            x = self.dropout(x)
            return x

    mod = torch.jit.script(DropoutModel())
    mod = torch.jit.freeze(mod.eval())
    list_of_inputs.append({"mod": mod})

    # Example 5: Conv2d module
    class ConvModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, kernel_size=3)
            self.relu = torch.nn.ReLU()
            self.pool = torch.nn.MaxPool2d(2, 2)

        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            x = self.pool(x)
            return x

    mod = torch.jit.script(ConvModel())
    mod = torch.jit.freeze(mod.eval())
    list_of_inputs.append({"mod": mod})

    # Example 6: Script module with if-else statement
    @torch.jit.script
    def scripted_fn(x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 0:
            return x * 2
        else:
            return x / 2

    class IfElseModel(torch.nn.Module):
        def forward(self, x):
            return scripted_fn(x)

    mod = torch.jit.script(IfElseModel())
    mod = torch.jit.freeze(mod.eval())
    list_of_inputs.append({"mod": mod})

    return list_of_inputs

generated_inputs = optimize_for_inference_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('optimize_for_inference', generated_inputs)
