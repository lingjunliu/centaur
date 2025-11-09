
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_UnicodeScript_inputs():
    list_of_inputs = []
    
    input_tensor = np.array([65, 66, 67], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([1, 31, 38], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[65, 945, 1040], [20013, 12354, 1488]], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([945], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([-1, -100, 65], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([0x1F600, 0x1F601, 0x1F602], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[[65, 66], [67, 68]], [[945, 946], [947, 948]]], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([0, 32, 127], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([20013, 22269, 26085, 26412], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([1575, 1576, 1577, 1578], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([1488, 1489, 1490], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.UnicodeScript"] = tf_raw_ops_UnicodeScript_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.UnicodeScript' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.UnicodeScript'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.UnicodeScript', generated_inputs['tf.raw_ops.UnicodeScript'], lib="tf", suffix=0)
