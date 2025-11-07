
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_moveaxis_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    a = np.array([[1, 2, 3], [4, 5, 6]])
    source = 0
    destination = 1
    input_dict = {
        "a": a,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    source = -1
    destination = 0
    input_dict = {
        "a": a,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    source = 0
    destination = -1
    input_dict = {
        "a": a,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    source = -2
    destination = -1
    input_dict = {
        "a": a,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    source = -1
    destination = 0
    input_dict = {
        "a": a,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    source = 1
    destination = -1
    input_dict = {
        "a": a,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    source = 0
    destination = -2
    input_dict = {
        "a": a,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    source = -1
    destination = 1
    input_dict = {
        "a": a,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    source = 0
    destination = -1
    input_dict = {
        "a": a,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    source = -2
    destination = 0
    input_dict = {
        "a": a,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.moveaxis"] = tf_moveaxis_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.moveaxis' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.moveaxis'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.moveaxis', generated_inputs['tf.experimental.numpy.moveaxis'], lib="tf", suffix=0)
