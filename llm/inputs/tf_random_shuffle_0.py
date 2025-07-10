
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_random_shuffle_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D tensor
    value = np.array([[1, 2], [3, 4], [5, 6]])
    seed = 123
    name = "shuffle_1"
    input_dict = {"value": value, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D tensor
    value = np.array([1, 2, 3, 4, 5])
    seed = 42
    name = "shuffle_2"
    input_dict = {"value": value, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor
    value = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    seed = 7
    name = "shuffle_3"
    input_dict = {"value": value, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tensor with negative values
    value = np.array([[-1, -2], [-3, -4], [-5, -6]])
    seed = 99
    name = "shuffle_4"
    input_dict = {"value": value, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with zeros
    value = np.array([[0, 0], [0, 0], [0, 0]])
    seed = 1
    name = "shuffle_5"
    input_dict = {"value": value, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger tensor
    value = np.random.rand(100, 5)
    seed = 55
    name = "shuffle_6"
    input_dict = {"value": value, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensor with different data type (int)
    value = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    seed = 22
    name = "shuffle_7"
    input_dict = {"value": value, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Empty Tensor
    value = np.array([])
    seed = 23
    name = "shuffle_8"
    input_dict = {"value": value, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Another 3D Tensor
    value = np.random.rand(5, 3, 2)
    seed = 10
    name = "shuffle_9"
    input_dict = {"value": value, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Seed as 0
    value = np.array([[7, 8], [9, 10], [11, 12]])
    seed = 0
    name = "shuffle_10"
    input_dict = {"value": value, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.random.shuffle"] = tf_random_shuffle_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.random.shuffle' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.shuffle'.")

check_valid('tf.random.shuffle', generated_inputs['tf.random.shuffle'], lib="tf", suffix=0)
