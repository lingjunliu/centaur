
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_experimental_numpy_random_randn_inputs():
    list_of_inputs = []

    # Input 1: Scalar shape
    input_dict = {"args": (np.array(1, dtype=np.int64),)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D shape
    input_dict = {"args": (np.array(5, dtype=np.int64),)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D shape
    input_dict = {"args": (np.array(2, dtype=np.int64), np.array(3, dtype=np.int64))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D shape
    input_dict = {"args": (np.array(2, dtype=np.int64), np.array(3, dtype=np.int64), np.array(4, dtype=np.int64))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger shape
    input_dict = {"args": (np.array(10, dtype=np.int64), np.array(10, dtype=np.int64))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Shape with one dimension being 1
    input_dict = {"args": (np.array(1, dtype=np.int64), np.array(5, dtype=np.int64), np.array(1, dtype=np.int64))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7:  Empty tuple. Should return a scalar.
    input_dict = {"args": ()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Shape with zero dimension
    input_dict = {"args": (np.array(0, dtype=np.int64), np.array(5, dtype=np.int64))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Shape with large numbers
    input_dict = {"args": (np.array(100, dtype=np.int64), np.array(100, dtype=np.int64))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Shape with one large number
    input_dict = {"args": (np.array(1000, dtype=np.int64),)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

tf.random.set_seed(1)
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
