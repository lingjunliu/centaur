
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_invert_permutation_inputs():
    list_of_inputs = []

    x = np.array([3, 4, 0, 2, 1], dtype=np.int32)
    input_dict = {'x': x, 'name': 'example_1'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    input_dict = {'x': x, 'name': 'example_2'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([4, 3, 2, 1, 0], dtype=np.int32)
    input_dict = {'x': x, 'name': 'example_3'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1, 2, 0], dtype=np.int64)
    input_dict = {'x': x, 'name': 'example_4'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0], dtype=np.int32)
    input_dict = {'x': x, 'name': 'example_5'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([5, 0, 2, 1, 4, 3], dtype=np.int64)
    input_dict = {'x': x, 'name': 'example_6'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([2, 0, 1], dtype=np.int32)
    input_dict = {'x': x, 'name': 'example_7'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([10, 5, 0, 2, 8, 1, 7, 3, 6, 4, 9], dtype=np.int64)
    input_dict = {'x': x, 'name': 'example_8'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0, 2, 4, 1, 3], dtype=np.int32)
    input_dict = {'x': x, 'name': 'example_9'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([7, 6, 5, 4, 3, 2, 1, 0], dtype=np.int64)
    input_dict = {'x': x, 'name': 'example_10'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.invert_permutation"] = tf_math_invert_permutation_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.invert_permutation' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.invert_permutation'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.invert_permutation', generated_inputs['tf.math.invert_permutation'], lib="tf", suffix=0)
