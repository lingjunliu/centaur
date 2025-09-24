
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def rsub_inputs():
    list_of_inputs = []

    # Test case 1: Basic float tensors
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    other1 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    alpha1 = 1.0
    input_dict1 = {"input": input1, "other": other1, "alpha": alpha1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Test case 2: Integer tensors
    input2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    other2 = np.array([[5, 6], [7, 8]], dtype=np.int32)
    alpha2 = 2
    input_dict2 = {"input": input2, "other": other2, "alpha": alpha2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Test case 3: Negative values
    input3 = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    other3 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    alpha3 = 1.0
    input_dict3 = {"input": input3, "other": other3, "alpha": alpha3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Test case 4: Different shapes (1D tensors)
    input4 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    other4 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    alpha4 = 0.5
    input_dict4 = {"input": input4, "other": other4, "alpha": alpha4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Test case 5: Different shapes (3D tensors)
    input5 = np.random.rand(2, 3, 4).astype(np.float32)
    other5 = np.random.rand(2, 3, 4).astype(np.float32)
    alpha5 = 1.5
    input_dict5 = {"input": input5, "other": other5, "alpha": alpha5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Test case 6: Scalar other
    input6 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    other6 = np.array(5.0, dtype=np.float32)
    alpha6 = 1.0
    input_dict6 = {"input": input6, "other": other6, "alpha": alpha6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Test case 7: Scalar input, tensor other
    input7 = np.array(2.0, dtype=np.float32)
    other7 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    alpha7 = 1.0
    input_dict7 = {"input": input7, "other": other7, "alpha": alpha7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Test case 8: Complex tensors
    input8 = np.array([[1.0 + 1j, 2.0 - 2j], [3.0, 4.0 + 1j]], dtype=np.complex64)
    other8 = np.array([[5.0, 6.0], [7.0 - 1j, 8.0]], dtype=np.complex64)
    alpha8 = 1.0
    input_dict8 = {"input": input8, "other": other8, "alpha": alpha8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    # Test case 9: Integer tensors with alpha as integer
    input9 = np.array([[1, 2], [3, 4]], dtype=np.int64)
    other9 = np.array([[5, 6], [7, 8]], dtype=np.int64)
    alpha9 = 2
    input_dict9 = {"input": input9, "other": other9, "alpha": alpha9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    return list_of_inputs

generated_inputs = rsub_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('rsub', generated_inputs)
