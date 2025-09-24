
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def arccos__inputs():
    generated_inputs = []

    # Input 1: Basic float tensor within [-1, 1]
    input1 = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    generated_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Multi-dimensional float tensor
    input2 = np.array([[-0.8, 0.2], [0.7, -0.3]], dtype=np.float64)
    input_dict2 = {"input": input2}
    generated_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Float tensor with values outside [-1, 1] (will result in NaN)
    input3 = np.array([-2.0, 1.5, 0.0], dtype=np.float32)
    input_dict3 = {"input": input3}
    generated_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Float tensor with a single large dimension
    input4 = np.random.uniform(low=-1.0, high=1.0, size=(1, 1000)).astype(np.float32)
    input_dict4 = {"input": input4}
    generated_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Scalar input
    input5 = np.array(0.6, dtype=np.float32)
    input_dict5 = {"input": input5}
    generated_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: Negative values close to -1
    input6 = np.array([-0.99, -0.999, -0.9999], dtype=np.float32)
    input_dict6 = {"input": input6}
    generated_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Positive values close to 1
    input7 = np.array([0.99, 0.999, 0.9999], dtype=np.float32)
    input_dict7 = {"input": input7}
    generated_inputs.append(copy.deepcopy(input_dict7))
    
    return generated_inputs

generated_inputs = arccos__inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('arccos_', generated_inputs)
