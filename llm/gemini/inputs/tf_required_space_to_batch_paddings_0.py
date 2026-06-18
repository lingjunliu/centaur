
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_required_space_to_batch_paddings_inputs():
    list_of_inputs = []
    
    # Input 1
    input_dict = {
        'input_shape': np.array([10], dtype=np.int32),
        'block_shape': np.array([3], dtype=np.int32),
        'base_paddings': np.array([[1, 2]], dtype=np.int32),
        'name': "test1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    input_dict = {
        'input_shape': np.array([10, 20], dtype=np.int32),
        'block_shape': np.array([3, 4], dtype=np.int32),
        'base_paddings': np.array([[0, 0], [1, 1]], dtype=np.int32),
        'name': "test2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    input_dict = {
        'input_shape': np.array([14, 15, 16], dtype=np.int32),
        'block_shape': np.array([2, 3, 4], dtype=np.int32),
        'base_paddings': np.array([[0, 1], [2, 0], [1, 1]], dtype=np.int32),
        'name': "test3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input_dict = {
        'input_shape': np.array([5], dtype=np.int32),
        'block_shape': np.array([5], dtype=np.int32),
        'base_paddings': np.array([[0, 0]], dtype=np.int32),
        'name': "test4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input_dict = {
        'input_shape': np.array([100, 200, 300, 400], dtype=np.int32),
        'block_shape': np.array([7, 8, 9, 10], dtype=np.int32),
        'base_paddings': np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int32),
        'name': "test5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    input_dict = {
        'input_shape': np.array([1, 1], dtype=np.int32),
        'block_shape': np.array([10, 10], dtype=np.int32),
        'base_paddings': np.array([[5, 5], [2, 3]], dtype=np.int32),
        'name': "test6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input_dict = {
        'input_shape': np.array([1000], dtype=np.int32),
        'block_shape': np.array([1], dtype=np.int32),
        'base_paddings': np.array([[0, 0]], dtype=np.int32),
        'name': "test7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input_dict = {
        'input_shape': np.array([7, 11, 13], dtype=np.int32),
        'block_shape': np.array([5, 5, 5], dtype=np.int32),
        'base_paddings': np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32),
        'name': "test8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_dict = {
        'input_shape': np.array([100, 100], dtype=np.int32),
        'block_shape': np.array([99, 99], dtype=np.int32),
        'base_paddings': np.array([[10, 10], [20, 20]], dtype=np.int32),
        'name': "test9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input_dict = {
        'input_shape': np.array([2, 4, 6, 8, 10], dtype=np.int32),
        'block_shape': np.array([3, 3, 3, 3, 3], dtype=np.int32),
        'base_paddings': np.array([[1, 1], [2, 2], [0, 0], [1, 2], [2, 1]], dtype=np.int32),
        'name': "test10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.required_space_to_batch_paddings"] = tf_required_space_to_batch_paddings_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.required_space_to_batch_paddings' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.required_space_to_batch_paddings'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.required_space_to_batch_paddings', generated_inputs['tf.required_space_to_batch_paddings'], lib="tf", suffix=0)
