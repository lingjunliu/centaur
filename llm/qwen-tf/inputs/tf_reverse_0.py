
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_reverse_inputs():
    list_of_inputs = []
    
    # Input 1: Reverse last dimension (axis = [3])
    tensor_1 = np.array([[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]], dtype=np.int32)
    axis_1 = np.array([3], dtype=np.int32)
    input_dict = {
        "tensor": tensor_1,
        "axis": axis_1,
        "name": "reverse_last_dim"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Reverse first dimension (axis = [0])
    tensor_2 = np.array([[[[0, 1], [2, 3]], [[4, 5], [6, 7]]]], dtype=np.int32)
    axis_2 = np.array([0], dtype=np.int32)
    input_dict = {
        "tensor": tensor_2,
        "axis": axis_2,
        "name": "reverse_first_dim"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Reverse middle dimension (axis = [1])
    tensor_3 = np.array([[[[0, 1], [2, 3]], [[4, 5], [6, 7]]]], dtype=np.int32)
    axis_3 = np.array([1], dtype=np.int32)
    input_dict = {
        "tensor": tensor_3,
        "axis": axis_3,
        "name": "reverse_middle_dim"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Reverse last two dimensions (axis = [2, 3])
    tensor_4 = np.array([[[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]]], dtype=np.int32)
    axis_4 = np.array([2, 3], dtype=np.int32)
    input_dict = {
        "tensor": tensor_4,
        "axis": axis_4,
        "name": "reverse_last_two_dims"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Reverse first two dimensions (axis = [0, 1])
    tensor_5 = np.array([[[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]]], dtype=np.int32)
    axis_5 = np.array([0, 1], dtype=np.int32)
    input_dict = {
        "tensor": tensor_5,
        "axis": axis_5,
        "name": "reverse_first_two_dims"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Reverse negative indices (axis = [-1])
    tensor_6 = np.array([[[[0, 1], [2, 3]], [[4, 5], [6, 7]]]], dtype=np.int32)
    axis_6 = np.array([-1], dtype=np.int32)
    input_dict = {
        "tensor": tensor_6,
        "axis": axis_6,
        "name": "reverse_negative_index"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Reverse negative indices (axis = [-2])
    tensor_7 = np.array([[[[0, 1], [2, 3]], [[4, 5], [6, 7]]]], dtype=np.int32)
    axis_7 = np.array([-2], dtype=np.int32)
    input_dict = {
        "tensor": tensor_7,
        "axis": axis_7,
        "name": "reverse_negative_index_neg"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Reverse multiple dimensions with negative indices (axis = [-1, -2])
    tensor_8 = np.array([[[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]]], dtype=np.int32)
    axis_8 = np.array([-1, -2], dtype=np.int32)
    input_dict = {
        "tensor": tensor_8,
        "axis": axis_8,
        "name": "reverse_multiple_negatives"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Reverse zero dimensions (axis = []) with 1D tensor
    tensor_9 = np.array([[0, 1, 2, 3]], dtype=np.int32)
    axis_9 = np.array([], dtype=np.int32)
    input_dict = {
        "tensor": tensor_9,
        "axis": axis_9,
        "name": "reverse_zero_dims"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Reverse last dimension with string tensor
    tensor_10 = np.array([["a", "b", "c"], ["d", "e", "f"]], dtype=np.object_)
    axis_10 = np.array([1], dtype=np.int32)
    input_dict = {
        "tensor": tensor_10,
        "axis": axis_10,
        "name": "reverse_string"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.reverse"] = tf_reverse_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.reverse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.reverse'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.reverse', generated_inputs['tf.reverse'], lib="tf", suffix=0)
