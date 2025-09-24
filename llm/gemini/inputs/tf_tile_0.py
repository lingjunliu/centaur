
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_tile_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([1, 2, 3, 4])
    multiples_tensor = np.array([2])
    name = "tile_example_1"
    input_dict = {"input": input_tensor, "multiples": multiples_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[1, 2], [3, 4]])
    multiples_tensor = np.array([2, 3])
    name = "tile_example_2"
    input_dict = {"input": input_tensor, "multiples": multiples_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    multiples_tensor = np.array([1, 2, 1])
    name = "tile_example_3"
    input_dict = {"input": input_tensor, "multiples": multiples_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([1, 2, 3])
    multiples_tensor = np.array([3])
    name = "tile_example_4"
    input_dict = {"input": input_tensor, "multiples": multiples_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]])
    multiples_tensor = np.array([1, 1])
    name = "tile_example_5"
    input_dict = {"input": input_tensor, "multiples": multiples_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([1])
    multiples_tensor = np.array([5])
    name = "tile_example_6"
    input_dict = {"input": input_tensor, "multiples": multiples_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([[1]])
    multiples_tensor = np.array([3, 2])
    name = "tile_example_7"
    input_dict = {"input": input_tensor, "multiples": multiples_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([1, 2])
    multiples_tensor = np.array([4])
    name = "tile_example_8"
    input_dict = {"input": input_tensor, "multiples": multiples_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([[[1]]])
    multiples_tensor = np.array([2, 3, 4])
    name = "tile_example_9"
    input_dict = {"input": input_tensor, "multiples": multiples_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([1, 2, 3, 4, 5])
    multiples_tensor = np.array([1])
    name = "tile_example_10"
    input_dict = {"input": input_tensor, "multiples": multiples_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.tile"] = tf_tile_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.tile' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.tile'.")

check_valid('tf.tile', generated_inputs['tf.tile'], lib="tf", suffix=0)
