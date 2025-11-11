
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_nextafter_inputs():
    list_of_inputs = []
    
    # Input 1: float32 tensor with positive values
    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x2 = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "name": "test1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: float64 tensor with negative values
    x1 = np.array([-1.0, -2.0], dtype=np.float64)
    x2 = np.array([-2.0, -3.0], dtype=np.float64)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "name": "test2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: float32 tensor with zero values
    x1 = np.array([0.0, 0.0], dtype=np.float32)
    x2 = np.array([1e-45, 1e-45], dtype=np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "name": "test3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: float64 tensor with subnormal values
    x1 = np.array([1e-30, 1e-30], dtype=np.float64)
    x2 = np.array([1e-31, 1e-31], dtype=np.float64)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "name": "test4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: float32 tensor with mixed values
    x1 = np.array([1.5, 2.7, 3.1], dtype=np.float32)
    x2 = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "name": "test5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: float64 tensor with single element
    x1 = np.array([1.0], dtype=np.float64)
    x2 = np.array([2.0], dtype=np.float64)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "name": "test6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: float32 tensor with high precision values
    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "name": "test7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: float64 tensor with different dimensions
    x1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    x2 = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float64)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "name": "test8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: float32 tensor with nan values
    x1 = np.array([np.nan, 1.0], dtype=np.float32)
    x2 = np.array([1.0, 2.0], dtype=np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "name": "test9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: float64 tensor with large values
    x1 = np.array([1e30, 2e30], dtype=np.float64)
    x2 = np.array([2e30, 3e30], dtype=np.float64)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "name": "test10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.nextafter"] = tf_math_nextafter_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.nextafter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.nextafter'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.nextafter', generated_inputs['tf.math.nextafter'], lib="tf", suffix=0)
