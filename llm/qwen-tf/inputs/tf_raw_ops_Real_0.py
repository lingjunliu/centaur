
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_real_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input_tensor = np.array([-2.25 + 4.75j, 3.25 + 5.75j], dtype=np.complex64)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float32,
        "name": "real1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input_tensor = np.array([1.5 + 2.5j], dtype=np.complex64)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float32,
        "name": "real2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input_tensor = np.array([[1.0 + 2.0j, 3.0 + 4.0j], [5.0 + 6.0j, 7.0 + 8.0j]], dtype=np.complex64)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float32,
        "name": "real3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input_tensor = np.array([0.0 + 0.0j], dtype=np.complex64)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float32,
        "name": "real4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input_tensor = np.array([-1.0 - 1.0j], dtype=np.complex64)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float32,
        "name": "real5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input_tensor = np.array([1.0 + 1.0j, -1.0 - 1.0j], dtype=np.complex64)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float32,
        "name": "real6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input_tensor = np.array([1.0 + 1.0j, -1.0 - 1.0j], dtype=np.complex128)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float64,
        "name": "real7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input_tensor = np.array([0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float64,
        "name": "real8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input_tensor = np.array([0.5 + 1.5j], dtype=np.complex64)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float32,
        "name": "real9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input_tensor = np.array([-3.0 - 4.0j], dtype=np.complex64)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float32,
        "name": "real10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Real"] = tf_real_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Real' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Real'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Real', generated_inputs['tf.raw_ops.Real'], lib="tf", suffix=0)
