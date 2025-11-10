
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_logical_not_inputs():
    list_of_inputs = []
    
    # Input 1: Simple 1D array with True/False values
    x = np.array([True, False], dtype=bool)
    input_dict = {"x": x, "name": "logical_not_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: All True values
    x = np.array([True, True, True], dtype=bool)
    input_dict = {"x": x, "name": "logical_not_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: All False values
    x = np.array([False, False, False, False], dtype=bool)
    input_dict = {"x": x, "name": "logical_not_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 2D array
    x = np.array([[True, False], [False, True]], dtype=bool)
    input_dict = {"x": x, "name": "logical_not_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 3D array
    x = np.array([[[True, False], [True, True]], [[False, False], [True, False]]], dtype=bool)
    input_dict = {"x": x, "name": "logical_not_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Single scalar value (0D)
    x = np.array(True, dtype=bool)
    input_dict = {"x": x, "name": "logical_not_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Single scalar False
    x = np.array(False, dtype=bool)
    input_dict = {"x": x, "name": "logical_not_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Larger 1D array
    x = np.array([True, False, True, False, True, False, True, False], dtype=bool)
    input_dict = {"x": x, "name": "logical_not_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 2D array with different shape
    x = np.array([[True, True, True], [False, False, False]], dtype=bool)
    input_dict = {"x": x, "name": "logical_not_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 4D array
    x = np.array([[[[True, False], [False, True]]]], dtype=bool)
    input_dict = {"x": x, "name": "logical_not_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Large 2D array
    x = np.array([[True, False, True, False], [False, True, False, True], [True, True, False, False]], dtype=bool)
    input_dict = {"x": x, "name": "logical_not_11"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: Empty name parameter
    x = np.array([True, False], dtype=bool)
    input_dict = {"x": x, "name": ""}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.logical_not"] = tf_math_logical_not_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.logical_not' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.logical_not'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.logical_not', generated_inputs['tf.math.logical_not'], lib="tf", suffix=0)
