
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def less_inputs():
    generated_inputs = []

    # Test case 1: Basic float tensors
    input1 = torch.randn(2, 3).numpy()
    other1 = torch.randn(2, 3).numpy()
    generated_inputs.append({"input": input1, "other": other1})

    # Test case 2: Integer tensors
    input2 = torch.randint(-5, 5, (4, 4)).numpy()
    other2 = torch.randint(-5, 5, (4, 4)).numpy()
    generated_inputs.append({"input": input2, "other": other2})

    # Test case 3: Different shapes (but broadcastable)
    input3 = torch.randn(1, 5).numpy()
    other3 = torch.randn(5).numpy()
    generated_inputs.append({"input": input3, "other": other3})

    # Test case 4: Scalar comparison
    input4 = torch.randn(3, 3).numpy()
    other4 = np.float64(0.5)
    generated_inputs.append({"input": input4, "other": other4})

    # Test case 5: Scalar comparison with integer tensor
    input5 = torch.randint(-10, 10, (2, 2)).numpy()
    other5 = np.int32(3)
    generated_inputs.append({"input": input5, "other": other5})
    
    # Test case 6: Higher dimensions
    input6 = torch.randn(2, 3, 4).numpy()
    other6 = torch.randn(2, 3, 4).numpy()
    generated_inputs.append({"input": input6, "other": other6})

    # Test case 7: Negative values
    input7 = torch.randn(5, 5) - 2.0
    other7 = torch.randn(5, 5) - 1.0
    generated_inputs.append({"input": input7.numpy(), "other": other7.numpy()})

    # Test case 8: zero values
    input8 = torch.zeros(2, 2).numpy()
    other8 = torch.ones(2, 2).numpy()
    generated_inputs.append({"input": input8, "other": other8})

    # Test case 9: same values
    input9 = torch.ones(3, 3).numpy()
    other9 = torch.ones(3, 3).numpy()
    generated_inputs.append({"input": input9, "other": other9})
    
    return generated_inputs

generated_inputs = less_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('less', generated_inputs)
