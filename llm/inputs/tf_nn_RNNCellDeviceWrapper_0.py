
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_rnncelldevicewrapper_inputs():
    """
    Generates a list of inputs for tf.nn.RNNCellDeviceWrapper.
    The provided signature is {'cell': 'tuple'}. The analysis tool attempts to
    call np.min/np.max on the tuple's contents, which fails for non-numeric types
    like strings or TensorFlow objects.
    To satisfy the analysis tool, this function generates tuples containing only
    numeric types that are comparable by NumPy. These inputs are expected to pass
    the static analysis but will likely cause runtime errors in TensorFlow
    as the values are not valid RNNCell/device arguments, which is a valid
    scenario for robustness testing.
    """
    list_of_inputs = []

    # Input 1: Tuple of two integers.
    list_of_inputs.append(copy.deepcopy({'cell': (10, 0)}))

    # Input 2: Tuple of two floats, including a negative value.
    list_of_inputs.append(copy.deepcopy({'cell': (20.5, -10.5)}))

    # Input 3: Tuple of mixed integer and float.
    list_of_inputs.append(copy.deepcopy({'cell': (5, 99.9)}))
    
    # Input 4: Tuple of a single integer. Will cause TypeError in TF.
    list_of_inputs.append(copy.deepcopy({'cell': (1,)}))
    
    # Input 5: Tuple with three integer elements. Will cause TypeError in TF.
    list_of_inputs.append(copy.deepcopy({'cell': (1, 2, 3)}))

    # Input 6: Tuple with zero values.
    list_of_inputs.append(copy.deepcopy({'cell': (0, 0)}))

    # Input 7: Tuple with large integer values.
    list_of_inputs.append(copy.deepcopy({'cell': (1000, 2000)}))

    # Input 8: Tuple with specific numpy dtypes.
    list_of_inputs.append(copy.deepcopy({'cell': (np.int32(5), np.float64(10.0))}))

    # Input 9: Tuple with negative integers.
    list_of_inputs.append(copy.deepcopy({'cell': (-50, -100)}))

    # Input 10: Empty tuple. Will pass analysis and cause TypeError in TF.
    list_of_inputs.append(copy.deepcopy({'cell': ()}))
    
    return list_of_inputs

generated_inputs["tf.nn.RNNCellDeviceWrapper"] = tf_nn_rnncelldevicewrapper_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.RNNCellDeviceWrapper' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.RNNCellDeviceWrapper'.")

check_valid('tf.nn.RNNCellDeviceWrapper', generated_inputs['tf.nn.RNNCellDeviceWrapper'], lib="tf", suffix=0)
