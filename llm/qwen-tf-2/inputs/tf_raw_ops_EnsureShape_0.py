
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf

def tf_raw_ops_ensureshape_inputs():
    list_of_inputs = []
    
    # Input 1: 2D tensor with shape [2, 3]
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    shape_list = [2, 3]
    
    input_dict = {
        "name": "test1",
        "input": input_tensor,
        "shape": shape_list
    }
    list_of_inputs.append(input_dict)
    
    # Input 2: 3D tensor with shape [2, 2, 3]
    input_tensor = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [0, 1, 2]]], dtype=np.float32)
    shape_list = [2, 2, 3]
    
    input_dict = {
        "name": "test2",
        "input": input_tensor,
        "shape": shape_list
    }
    list_of_inputs.append(input_dict)
    
    # Input 3: 1D tensor with shape [5]
    input_tensor = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    shape_list = [5]
    
    input_dict = {
        "name": "test3",
        "input": input_tensor,
        "shape": shape_list
    }
    list_of_inputs.append(input_dict)
    
    # Input 4: 0D tensor (scalar) with shape []
    input_tensor = np.array(42, dtype=np.int32)
    shape_list = []
    
    input_dict = {
        "name": "test4",
        "input": input_tensor,
        "shape": shape_list
    }
    list_of_inputs.append(input_dict)
    
    # Input 5: 1D tensor with shape [3] (with negative values)
    input_tensor = np.array([-1, -2, -3], dtype=np.float64)
    shape_list = [3]
    
    input_dict = {
        "name": "test5",
        "input": input_tensor,
        "shape": shape_list
    }
    list_of_inputs.append(input_dict)
    
    # Input 6: 4D tensor with shape [1, 2, 3, 4]
    input_tensor = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 0, 1, 2], [3, 4, 5, 6]]]], dtype=np.int32)
    shape_list = [1, 2, 3, 4]
    
    input_dict = {
        "name": "test6",
        "input": input_tensor,
        "shape": shape_list
    }
    list_of_inputs.append(input_dict)
    
    # Input 7: 2D tensor with shape [3, 3] (square matrix)
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    shape_list = [3, 3]
    
    input_dict = {
        "name": "test7",
        "input": input_tensor,
        "shape": shape_list
    }
    list_of_inputs.append(input_dict)
    
    # Input 8: 1D tensor with shape [10] (large size)
    input_tensor = np.array(range(10), dtype=np.int32)
    shape_list = [10]
    
    input_dict = {
        "name": "test8",
        "input": input_tensor,
        "shape": shape_list
    }
    list_of_inputs.append(input_dict)
    
    # Input 9: 2D tensor with shape [4, 5] (different dimensions)
    input_tensor = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]], dtype=np.float64)
    shape_list = [4, 5]
    
    input_dict = {
        "name": "test9",
        "input": input_tensor,
        "shape": shape_list
    }
    list_of_inputs.append(input_dict)
    
    # Input 10: 3D tensor with shape [2, 3, 4] (mixed dimensions)
    input_tensor = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 1, 2]], [[3, 4, 5, 6], [7, 8, 9, 0], [1, 2, 3, 4]]], dtype=np.int64)
    shape_list = [2, 3, 4]
    
    input_dict = {
        "name": "test10",
        "input": input_tensor,
        "shape": shape_list
    }
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs["tf.raw_ops.EnsureShape"] = tf_raw_ops_ensureshape_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.EnsureShape' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.EnsureShape'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.EnsureShape', generated_inputs['tf.raw_ops.EnsureShape'], lib="tf", suffix=0)
