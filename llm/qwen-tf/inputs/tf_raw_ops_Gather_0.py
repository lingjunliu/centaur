
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_Gather_inputs():
    list_of_inputs = []
    
    # Input 1: scalar indices, 2D params
    params = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    indices = np.array(1, dtype=np.int32)
    validate_indices = True
    name = "gather_1"
    
    input_dict = {
        "validate_indices": validate_indices,
        "name": name,
        "params": params,
        "indices": indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: vector indices, 3D params
    params = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    indices = np.array([0, 1], dtype=np.int32)
    validate_indices = True
    name = "gather_2"
    
    input_dict = {
        "validate_indices": validate_indices,
        "name": name,
        "params": params,
        "indices": indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: higher rank indices, 4D params
    params = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]], dtype=np.int32)
    indices = np.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]], [[0, 1], [1, 0]]], dtype=np.int32)
    validate_indices = False
    name = "gather_3"
    
    input_dict = {
        "validate_indices": validate_indices,
        "name": name,
        "params": params,
        "indices": indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: vector index, 1D params
    params = np.array([1, 2, 3], dtype=np.int32)
    indices = np.array([0, 1, 2], dtype=np.int32)
    validate_indices = True
    name = "gather_4"
    
    input_dict = {
        "validate_indices": validate_indices,
        "name": name,
        "params": params,
        "indices": indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: scalar index, 1D params
    params = np.array([1, 2, 3], dtype=np.int32)
    indices = np.array(0, dtype=np.int32)
    validate_indices = True
    name = "gather_5"
    
    input_dict = {
        "validate_indices": validate_indices,
        "name": name,
        "params": params,
        "indices": indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: scalar index, 3D params
    params = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    indices = np.array(0, dtype=np.int32)
    validate_indices = True
    name = "gather_6"
    
    input_dict = {
        "validate_indices": validate_indices,
        "name": name,
        "params": params,
        "indices": indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: vector index, 3D params
    params = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    indices = np.array([1, 0], dtype=np.int32)
    validate_indices = False
    name = "gather_7"
    
    input_dict = {
        "validate_indices": validate_indices,
        "name": name,
        "params": params,
        "indices": indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: scalar index, 1D params with negative value
    params = np.array([1, 2, 3], dtype=np.int32)
    indices = np.array(-1, dtype=np.int32)
    validate_indices = False
    name = "gather_8"
    
    input_dict = {
        "validate_indices": validate_indices,
        "name": name,
        "params": params,
        "indices": indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: vector index, 2D params with negative values
    params = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    indices = np.array([0, -1], dtype=np.int32)
    validate_indices = True
    name = "gather_9"
    
    input_dict = {
        "validate_indices": validate_indices,
        "name": name,
        "params": params,
        "indices": indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: vector index, 1D params with negative values
    params = np.array([1, 2, 3], dtype=np.int32)
    indices = np.array([-1, -2], dtype=np.int32)
    validate_indices = True
    name = "gather_10"
    
    input_dict = {
        "validate_indices": validate_indices,
        "name": name,
        "params": params,
        "indices": indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Gather"] = tf_raw_ops_Gather_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Gather' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Gather'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Gather', generated_inputs['tf.raw_ops.Gather'], lib="tf", suffix=0)
