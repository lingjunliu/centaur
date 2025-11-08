
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_dot_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    b = np.array([[2, 4, -6], [5, 7, 9]], dtype=np.int32)
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    b = np.array([[[2, 3], [4, 5]], [[6, 7], [8, 9]]], dtype=np.float32)
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    a = np.array([1, 2, 3], dtype=np.int64)
    b = np.array([4, 5, 6], dtype=np.int64)
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    a = np.array([[1, 2], [3, 4]], dtype=np.float64)
    b = np.array([[-1, 0], [1, 2]], dtype=np.float64)
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    a = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [0, 1, 2]]], dtype=np.float32)
    b = np.array([[[1, 0], [2, 3]], [[4, 5], [6, 7]]], dtype=np.float32)
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    b = np.array([[-1, 0, 1], [2, 3, 4], [5, 6, 7]], dtype=np.int32)
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    a = np.array([1, 2, 3], dtype=np.float64)
    b = np.array([-1, 0, 1], dtype=np.float64)
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    a = np.array([[1, 2], [3, 4]], dtype=np.int64)
    b = np.array([[0, -1], [1, 2]], dtype=np.int64)
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    b = np.array([[1, 2], [3, 4]], dtype=np.float32)
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    a = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [0, 1, 2]]], dtype=np.int32)
    b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.dot"] = tf_dot_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.dot' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.dot'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.dot', generated_inputs['tf.experimental.numpy.dot'], lib="tf", suffix=0)
