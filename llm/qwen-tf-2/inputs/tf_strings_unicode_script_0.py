
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_unicode_script_inputs():
    list_of_inputs = []
    
    # Input 1: Simple 1D array
    input_tensor = np.array([65, 66, 67], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": "test1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 2D array
    input_tensor = np.array([[65, 66], [67, 68]], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": "test2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Single element array
    input_tensor = np.array([100], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": "test3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Negative values
    input_tensor = np.array([-65, -66, -67], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": "test4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Mixed values
    input_tensor = np.array([123, 456, 789], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": "test5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Large numbers
    input_tensor = np.array([1000, 2000, 3000], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": "test6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Empty array
    input_tensor = np.array([], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": "test7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 3D array
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": "test8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Negative and positive mixed
    input_tensor = np.array([100, -100, 200], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": "test9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Large array
    input_tensor = np.array([97, 98, 99, 100, 101], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": "test10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.strings.unicode_script"] = tf_strings_unicode_script_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.strings.unicode_script' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.unicode_script'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.strings.unicode_script', generated_inputs['tf.strings.unicode_script'], lib="tf", suffix=0)
