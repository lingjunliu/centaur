
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import torch.nn as nn
import copy
import numpy as np

def set_module_inputs():
    list_of_inputs = []

    class MyModule1(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)

    module1 = MyModule1()

    input_dict1 = {
        "name": "my_module1",
        "module": module1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    class MyModule2(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3)

        def forward(self, x):
            return self.conv(x)

    module2 = MyModule2()

    input_dict2 = {
        "name": "my_module2",
        "module": module2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    class MyModule3(nn.Module):
        def __init__(self):
            super().__init__()
            self.seq = nn.Sequential(
                nn.Linear(20, 10),
                nn.ReLU(),
                nn.Linear(10, 5)
            )

        def forward(self, x):
            return self.seq(x)

    module3 = MyModule3()
    input_dict3 = {
        "name": "my_module3",
        "module": module3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    class MyModule4(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(10, 20)

        def forward(self, x):
            out, _ = self.lstm(x)
            return out
    
    module4 = MyModule4()
    input_dict4 = {
        "name": "my_module4",
        "module": module4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    class MyModule5(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(1000, 128)

        def forward(self, x):
            return self.embedding(x)

    module5 = MyModule5()

    input_dict5 = {
        "name": "my_module5",
        "module": module5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    class MyModule6(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm1d(32)

        def forward(self, x):
            return self.bn(x)
    
    module6 = MyModule6()
    input_dict6 = {
        "name": "my_module6",
        "module": module6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    class MyModule7(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = nn.Dropout(p=0.5)

        def forward(self, x):
            return self.dropout(x)
    
    module7 = MyModule7()
    input_dict7 = {
        "name": "my_module7",
        "module": module7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = set_module_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('set_module', generated_inputs)
