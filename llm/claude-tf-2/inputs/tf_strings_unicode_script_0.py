
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_strings_unicode_script_inputs():
    list_of_inputs = []
    
    input_dict = {
        "input": np.array([65, 66, 67], dtype=np.int32),
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([0x0041, 0x0410, 0x4E00], dtype=np.int32),
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[72, 101, 108], [108, 111, 33]], dtype=np.int32),
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[[48, 49], [50, 51]], [[52, 53], [54, 55]]], dtype=np.int32),
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([97], dtype=np.int32),
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([-1, -5, -100], dtype=np.int32),
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([65, -1, 0x4E00, 0], dtype=np.int32),
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([0x1F600, 0x1F601, 0x1F602], dtype=np.int32),
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([0, 0, 0], dtype=np.int32),
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([0x0041, 0x0410, 0x0627, 0x05D0], dtype=np.int32),
        "name": None
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
