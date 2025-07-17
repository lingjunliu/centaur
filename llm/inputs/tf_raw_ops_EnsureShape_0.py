
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_EnsureShape_inputs():
    list_of_inputs = []

    # Input 1: Simple case with matching shape
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.int32)
    shape = [2, 2]
    input_dict = {"input": input_tensor, "shape": shape, "name": "ensure_shape_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different data type
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    shape = [3]
    input_dict = {"input": input_tensor, "shape": shape, "name": "ensure_shape_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Higher dimensions
    input_tensor = np.random.rand(2, 3, 4).astype(np.float64)
    shape = [2, 3, 4]
    input_dict = {"input": input_tensor, "shape": shape, "name": "ensure_shape_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Scalar
    input_tensor = np.array(5, dtype=np.int64)
    shape = []
    input_dict = {"input": input_tensor, "shape": shape, "name": "ensure_shape_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Empty tensor
    input_tensor = np.array([], dtype=np.int32)
    shape = [0]
    input_dict = {"input": input_tensor, "shape": shape, "name": "ensure_shape_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D tensor
    input_tensor = np.array([1, 2, 3, 4, 5], dtype=np.int16)
    shape = [5]
    input_dict = {"input": input_tensor, "shape": shape, "name": "ensure_shape_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensor with negative values
    input_tensor = np.array([[-1, 2], [-3, 4]], dtype=np.int32)
    shape = [2, 2]
    input_dict = {"input": input_tensor, "shape": shape, "name": "ensure_shape_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tensor with zeros
    input_tensor = np.array([[0, 0], [0, 0]], dtype=np.int32)
    shape = [2, 2]
    input_dict = {"input": input_tensor, "shape": shape, "name": "ensure_shape_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9: Tensor with complex numbers
    input_tensor = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
    shape = [3]
    input_dict = {"input": input_tensor, "shape": shape, "name": "ensure_shape_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Boolean tensor
    input_tensor = np.array([[True, False], [False, True]], dtype=np.bool_)
    shape = [2, 2]
    input_dict = {"input": input_tensor, "shape": shape, "name": "ensure_shape_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.EnsureShape"] = tf_raw_ops_EnsureShape_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.EnsureShape' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.EnsureShape'.")

check_valid('tf.raw_ops.EnsureShape', generated_inputs['tf.raw_ops.EnsureShape'], lib="tf", suffix=0)
