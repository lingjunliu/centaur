
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_diag_part_inputs():
    list_of_inputs = []

    # Input 1: Basic square matrix
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Larger square matrix
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different data type (complex)
    input_tensor = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D tensor. Make sure dimensions are even and k is even.
    input_tensor = np.random.rand(2, 2, 2, 2).astype(np.float32)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different data type (bfloat16), requires casting from float32
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32).astype(np.float16)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Example with negative numbers
    input_tensor = np.array([[-1, 2], [3, -4]], dtype=np.int32)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Example with a name
    input_tensor = np.array([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]], dtype=np.int32)
    input_dict = {"input": input_tensor, "name": "my_diag_part"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Float64 type
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Half type
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Complex128 type
    input_tensor = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex128)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 6D Tensor (2,2,2,2,2,2)
    input_tensor = np.random.rand(2, 2, 2, 2, 2, 2).astype(np.float32)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: Empty name
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict = {"input": input_tensor, "name": ""}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DiagPart"] = tf_raw_ops_diag_part_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DiagPart' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DiagPart'.")

check_valid('tf.raw_ops.DiagPart', generated_inputs['tf.raw_ops.DiagPart'], lib="tf", suffix=0)
