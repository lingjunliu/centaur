
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf

def generate_expand_dims_inputs():
    list_of_inputs = []
    
    # Input 1: 2D array, axis=0
    a = np.array([[1, 2, 3], [4, 5, 6]])
    input_dict = {"a": a, "axis": 0}
    list_of_inputs.append(input_dict)
    
    # Input 2: 2D array, axis=1
    a = np.array([[1, 2, 3], [4, 5, 6]])
    input_dict = {"a": a, "axis": 1}
    list_of_inputs.append(input_dict)
    
    # Input 3: 1D array, axis=-1
    a = np.array([1, 2, 3])
    input_dict = {"a": a, "axis": -1}
    list_of_inputs.append(input_dict)
    
    # Input 4: 3D array, axis=0
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    input_dict = {"a": a, "axis": 0}
    list_of_inputs.append(input_dict)
    
    # Input 5: 3D array, axis=2
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    input_dict = {"a": a, "axis": 2}
    list_of_inputs.append(input_dict)
    
    # Input 6: 1D array, axis=0
    a = np.array([1, 2, 3])
    input_dict = {"a": a, "axis": 0}
    list_of_inputs.append(input_dict)
    
    # Input 7: 2D array, axis=1
    a = np.array([[1, 2, 3], [4, 5, 6]])
    input_dict = {"a": a, "axis": 1}
    list_of_inputs.append(input_dict)
    
    # Input 8: 3D array, axis=-2
    a = np.array([[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]])
    input_dict = {"a": a, "axis": -2}
    list_of_inputs.append(input_dict)
    
    # Input 9: 4D array, axis=-1
    a = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[9, 10], [11, 12]]])
    input_dict = {"a": a, "axis": -1}
    list_of_inputs.append(input_dict)
    
    # Input 10: 3D array, axis=1
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    input_dict = {"a": a, "axis": 1}
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs["tf.experimental.numpy.expand_dims"] = generate_expand_dims_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.expand_dims' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.expand_dims'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.expand_dims', generated_inputs['tf.experimental.numpy.expand_dims'], lib="tf", suffix=0)
