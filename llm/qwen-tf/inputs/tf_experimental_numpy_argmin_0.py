
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experiment_numpy_argmin_inputs():
    list_of_inputs = []
    
    # Input 1: 2D array with axis=0
    a = np.array([[1, 2, 3], [4, 5, 6]])
    input_dict = {"a": a, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 2D array with axis=1
    a = np.array([[1, 2, 3], [4, 5, 6]])
    input_dict = {"a": a, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 1D array
    a = np.array([1, 2, 3, 4])
    input_dict = {"a": a, "axis": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 3D array with axis=0
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    input_dict = {"a": a, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 3D array with axis=1
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    input_dict = {"a": a, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 3D array with axis=2
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    input_dict = {"a": a, "axis": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 1D array with negative values
    a = np.array([-1, 2, -3, 4])
    input_dict = {"a": a, "axis": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 2D array with negative values
    a = np.array([[-1, 2], [3, -4]])
    input_dict = {"a": a, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 2D array with negative values and axis=1
    a = np.array([[-1, 2], [3, -4]])
    input_dict = {"a": a, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 1D array with float values
    a = np.array([1.5, 2.7, 3.1])
    input_dict = {"a": a, "axis": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.argmin"] = tf_experiment_numpy_argmin_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.argmin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.argmin'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.argmin', generated_inputs['tf.experimental.numpy.argmin'], lib="tf", suffix=0)
