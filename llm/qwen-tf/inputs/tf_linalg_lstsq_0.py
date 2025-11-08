
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def lstsq_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    matrix = np.array([[1., 2.], [3., 4.], [5., 6.]])
    rhs = np.array([[7., 8.], [9., 10.], [11., 12.]])
    l2_regularizer = 0.0
    fast = True
    name = "test1"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "l2_regularizer": l2_regularizer,
        "fast": fast,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    matrix = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    rhs = np.array([[10., 11.], [12., 13.], [14., 15.]])
    l2_regularizer = 0.1
    fast = True
    name = "test2"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "l2_regularizer": l2_regularizer,
        "fast": fast,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    matrix = np.array([[1., 2.], [3., 4.], [5., 6.]])
    rhs = np.array([[7., 8., 9.], [10., 11., 12.]])
    l2_regularizer = 0.0
    fast = False
    name = "test3"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "l2_regularizer": l2_regularizer,
        "fast": fast,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    matrix = np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]])
    rhs = np.array([[9., 10.], [11., 12.]])
    l2_regularizer = 0.0
    fast = True
    name = "test4"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "l2_regularizer": l2_regularizer,
        "fast": fast,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    matrix = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    rhs = np.array([[10., 11., 12.], [13., 14., 15.], [16., 17., 18.]])
    l2_regularizer = 0.0
    fast = False
    name = "test5"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "l2_regularizer": l2_regularizer,
        "fast": fast,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    matrix = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    rhs = np.array([[10., 11., 12., 13.], [14., 15., 16., 17.], [18., 19., 20., 21.]])
    l2_regularizer = 0.0
    fast = True
    name = "test6"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "l2_regularizer": l2_regularizer,
        "fast": fast,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    matrix = np.array([[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.]])
    rhs = np.array([[11., 12., 13.], [14., 15., 16.], [17., 18., 19.], [20., 21., 22.]])
    l2_regularizer = 0.1
    fast = True
    name = "test7"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "l2_regularizer": l2_regularizer,
        "fast": fast,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    matrix = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    rhs = np.array([[10., 11., 12., 13., 14.], [15., 16., 17., 18., 19.], [20., 21., 22., 23., 24.]])
    l2_regularizer = 0.0
    fast = False
    name = "test8"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "l2_regularizer": l2_regularizer,
        "fast": fast,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    matrix = np.array([[1., 2., 3., 4., 5., 6.], [7., 8., 9., 10., 11., 12.]])
    rhs = np.array([[13., 14., 15., 16.], [17., 18., 19., 20.], [21., 22., 23., 24.], [25., 26., 27., 28.]])
    l2_regularizer = 0.0
    fast = True
    name = "test9"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "l2_regularizer": l2_regularizer,
        "fast": fast,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    matrix = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    rhs = np.array([[10., 11., 12., 13., 14.], [15., 16., 17., 18., 19.], [20., 21., 22., 23., 24.], [25., 26., 27., 28., 29.]])
    l2_regularizer = 0.0
    fast = False
    name = "test10"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "l2_regularizer": l2_regularizer,
        "fast": fast,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.linalg.lstsq"] = lstsq_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.lstsq' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.lstsq'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.linalg.lstsq', generated_inputs['tf.linalg.lstsq'], lib="tf", suffix=0)
