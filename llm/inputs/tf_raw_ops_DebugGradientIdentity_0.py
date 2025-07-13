
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_DebugGradientIdentity_inputs():
    list_of_inputs = []

    # Input 1: Float32 tensor, no name
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Int32 tensor, with name
    input_tensor = np.array([4, 5, 6], dtype=np.int32)
    input_dict = {"input": input_tensor, "name": "my_debug_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float64 tensor, negative values
    input_tensor = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Bool tensor
    input_tensor = np.array([True, False, True], dtype=np.bool_)
    input_dict = {"input": input_tensor, "name": "bool_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D Int64 tensor
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.int64)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Complex64 tensor
    input_tensor = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
    input_dict = {"input": input_tensor, "name": "complex_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: String tensor
    input_tensor = np.array(["hello", "world"], dtype=np.object_)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D Float16 tensor
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float16)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Uint8 tensor
    input_tensor = np.array([1, 2, 3], dtype=np.uint8)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty tensor
    input_tensor = np.array([], dtype=np.float32)
    input_dict = {"input": input_tensor, "name": "empty_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_DebugGradientIdentity_inputs()
generated_inputs["tf.raw_ops.DebugGradientIdentity"] = []
for input_dict in inputs:
    numpy_input = input_dict["input"]
    if numpy_input.size == 0:
        tensor_input = tf.constant(numpy_input, dtype=numpy_input.dtype)
    else:
        tensor_input = tf.convert_to_tensor(numpy_input)
    generated_inputs["tf.raw_ops.DebugGradientIdentity"].append({"input": tensor_input, "name": input_dict["name"]})

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DebugGradientIdentity' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DebugGradientIdentity'.")

check_valid('tf.raw_ops.DebugGradientIdentity', generated_inputs['tf.raw_ops.DebugGradientIdentity'], lib="tf", suffix=0)
