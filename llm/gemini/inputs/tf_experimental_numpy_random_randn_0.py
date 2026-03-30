
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_random_randn_inputs():
    list_of_inputs = []

    # Input 1: Empty shape
    input_dict = {"args": ()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Single integer
    input_dict = {"args": (np.int64(5),)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Two integers
    input_dict = {"args": (np.int64(2), np.int64(3))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Three integers
    input_dict = {"args": (np.int64(2), np.int64(3), np.int64(4))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Shape with 1
    input_dict = {"args": (np.int64(1),)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger shape
    input_dict = {"args": (np.int64(10), np.int64(10))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Shape with zero
    input_dict = {"args": (np.int64(0),)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Shape with a mix of large and small values
    input_dict = {"args": (np.int64(100), np.int64(1), np.int64(5))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Shape with a very large number
    input_dict = {"args": (np.int64(1000),)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Higher dimension shape
    input_dict = {"args": (np.int64(2), np.int64(2), np.int64(2), np.int64(2))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.random.randn"] = tf_experimental_numpy_random_randn_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.random.randn' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.random.randn'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.random.randn', generated_inputs['tf.experimental.numpy.random.randn'], lib="tf", suffix=0)
