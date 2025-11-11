
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_special_fresnel_cos_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    x = np.array([-1., -0.1, 0.1, 1.], dtype=np.float32)
    name = "test1"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    x = np.array([0.5, -0.5, 1.5, -1.5], dtype=np.float32)
    name = "test2"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    x = np.array([0.0], dtype=np.float32)
    name = "test3"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    x = np.array([-10., 10., -5., 5.], dtype=np.float32)
    name = "test4"

    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    x = np.array([0.99, -0.99, 1.99, -1.99], dtype=np.float32)
    name = "test5"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    name = "test6"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    x = np.array([[-1.0, -2.0], [3.0, -4.0]], dtype=np.float32)
    name = "test7"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    name = "test8"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    x = np.array([1e-5, -1e-5, 1e-3, -1e-3], dtype=np.float32)
    name = "test9"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    x = np.array([1.0, -1.0, 2.0, -2.0], dtype=np.float64)
    name = "test10"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.special.fresnel_cos"] = tf_math_special_fresnel_cos_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.special.fresnel_cos' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.special.fresnel_cos'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.special.fresnel_cos', generated_inputs['tf.math.special.fresnel_cos'], lib="tf", suffix=0)
