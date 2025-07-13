
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_DebugGradientRefIdentity_inputs():
    list_of_inputs = []

    def to_numpy(tensor):
        if isinstance(tensor, tf.Variable):
            return tensor.numpy()
        else:
            return tensor

    # Input 1: Simple float32 array
    input_tensor = tf.Variable(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    input_dict = {"input": input_tensor, "name": "float32_array"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Simple int32 array
    input_tensor = tf.Variable(np.array([1, 2, 3], dtype=np.int32))
    input_dict = {"input": input_tensor, "name": "int32_array"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 array
    input_tensor = tf.Variable(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
    input_dict = {"input": input_tensor, "name": "float64_2d_array"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D int64 array
    input_tensor = tf.Variable(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64))
    input_dict = {"input": input_tensor, "name": "int64_3d_array"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: boolean array
    input_tensor = tf.Variable(np.array([True, False, True], dtype=np.bool_))
    input_dict = {"input": input_tensor, "name": "bool_array"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Complex64 array
    input_tensor = tf.Variable(np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64))
    input_dict = {"input": input_tensor, "name": "complex64_array"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Complex128 array
    input_tensor = tf.Variable(np.array([1+1j, 2+2j, 3+3j], dtype=np.complex128))
    input_dict = {"input": input_tensor, "name": "complex128_array"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float16 array
    input_tensor = tf.Variable(np.array([1.0, 2.0, 3.0], dtype=np.float16))
    input_dict = {"input": input_tensor, "name": "float16_array"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: int8 array with negative values
    input_tensor = tf.Variable(np.array([-1, 0, 1], dtype=np.int8))
    input_dict = {"input": input_tensor, "name": "int8_array"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different name
    input_tensor = tf.Variable(np.array([5.0, 6.0, 7.0], dtype=np.float32))
    input_dict = {"input": input_tensor, "name": "another_name"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    for i in range(len(list_of_inputs)):
        list_of_inputs[i]["input"] = to_numpy(list_of_inputs[i]["input"])
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DebugGradientRefIdentity"] = tf_raw_ops_DebugGradientRefIdentity_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DebugGradientRefIdentity' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DebugGradientRefIdentity'.")

check_valid('tf.raw_ops.DebugGradientRefIdentity', generated_inputs['tf.raw_ops.DebugGradientRefIdentity'], lib="tf", suffix=0)
