
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Prod_inputs():
    list_of_inputs = []
    
    # Input 1: Simple 2D tensor with keep_dims=False
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    axis_tensor = np.array([1], dtype=np.int32)
    keep_dims = False
    name = "test1"
    
    input_dict = {
        "input": input_tensor,
        "axis": axis_tensor,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Simple 2D tensor with keep_dims=True
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.float64)
    axis_tensor = np.array([0], dtype=np.int32)
    keep_dims = True
    name = "test2"
    
    input_dict = {
        "input": input_tensor,
        "axis": axis_tensor,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 3D tensor with keep_dims=False
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    axis_tensor = np.array([0], dtype=np.int64)
    keep_dims = False
    name = "test3"
    
    input_dict = {
        "input": input_tensor,
        "axis": axis_tensor,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 3D tensor with keep_dims=True
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    axis_tensor = np.array([1], dtype=np.int32)
    keep_dims = True
    name = "test4"
    
    input_dict = {
        "input": input_tensor,
        "axis": axis_tensor,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 1D tensor with keep_dims=False
    input_tensor = np.array([1, 2, 3], dtype=np.float32)
    axis_tensor = np.array([0], dtype=np.int64)
    keep_dims = False
    name = "test5"
    
    input_dict = {
        "input": input_tensor,
        "axis": axis_tensor,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 4D tensor with negative axis values
    input_tensor = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]], dtype=np.float32)
    axis_tensor = np.array([-1], dtype=np.int32)
    keep_dims = False
    name = "test6"
    
    input_dict = {
        "input": input_tensor,
        "axis": axis_tensor,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 3D tensor with multiple axes
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    axis_tensor = np.array([0, 1], dtype=np.int64)
    keep_dims = False
    name = "test7"
    
    input_dict = {
        "input": input_tensor,
        "axis": axis_tensor,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 2D tensor with keep_dims=True and negative axis
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    axis_tensor = np.array([-1], dtype=np.int32)
    keep_dims = True
    name = "test8"
    
    input_dict = {
        "input": input_tensor,
        "axis": axis_tensor,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 3D tensor with all dimensions reduced
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int8)
    axis_tensor = np.array([0, 1, 2], dtype=np.int64)
    keep_dims = True
    name = "test9"
    
    input_dict = {
        "input": input_tensor,
        "axis": axis_tensor,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 2D tensor with complex numbers
    input_tensor = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex64)
    axis_tensor = np.array([1], dtype=np.int32)
    keep_dims = False
    name = "test10"
    
    input_dict = {
        "input": input_tensor,
        "axis": axis_tensor,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Prod"] = tf_raw_ops_Prod_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Prod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Prod'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Prod', generated_inputs['tf.raw_ops.Prod'], lib="tf", suffix=0)
