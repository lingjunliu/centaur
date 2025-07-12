
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.random.set_seed(1)

def tf_experimental_numpy_random_randn_inputs():
    list_of_inputs = []

    # Input 1: Scalar shape
    input_dict = {"args": (1,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D shape
    input_dict = {"args": (5,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D shape
    input_dict = {"args": (2, 3)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D shape
    input_dict = {"args": (2, 3, 4)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger shape
    input_dict = {"args": (10, 10)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Another 3D shape
    input_dict = {"args": (5, 2, 1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D shape
    input_dict = {"args": (2, 2, 2, 2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty tuple (results in scalar)
    input_dict = {"args": ()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large size
    input_dict = {"args": (100,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different shape
    input_dict = {"args": (3, 5, 2)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Shape as numpy array
    input_dict = {"args": (np.array([2, 3]),)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: Empty numpy array
    input_dict = {"args": (np.array([]),)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.random.randn"] = tf_experimental_numpy_random_randn_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.random.randn' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.random.randn'.")

check_valid('tf.experimental.numpy.random.randn', generated_inputs['tf.experimental.numpy.random.randn'], lib="tf", suffix=0)
