
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_dynamic_stitch_inputs():
    list_of_inputs = []

    # Input 1: Basic test case
    indices = [np.array([0, 2], dtype=np.int32), np.array([1, 3], dtype=np.int32)]
    data = [np.array([10, 12]), np.array([11, 13])]
    name = "stitch_example_1"
    input_dict = {"indices": indices, "data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different shapes for data, scalar indices
    indices = [np.array(0, dtype=np.int32), np.array(1, dtype=np.int32)]
    data = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    name = "stitch_example_2"
    input_dict = {"indices": indices, "data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Vector indices, 2D data
    indices = [np.array([0, 1], dtype=np.int32), np.array([2, 3], dtype=np.int32)]
    data = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    name = "stitch_example_3"
    input_dict = {"indices": indices, "data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D data, 2D indices - Corrected shape
    indices = [np.array([[0], [1]], dtype=np.int32), np.array([[2], [3]], dtype=np.int32)]
    data = [np.array([[[1]], [[2]]]), np.array([[[5]], [[6]]])]
    name = "stitch_example_4"
    input_dict = {"indices": indices, "data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Single index and data
    indices = [np.array([0, 1, 2], dtype=np.int32)]
    data = [np.array([1, 2, 3])]
    name = "stitch_example_5"
    input_dict = {"indices": indices, "data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Overlapping indices
    indices = [np.array([0, 1], dtype=np.int32), np.array([1, 2], dtype=np.int32)]
    data = [np.array([10, 11]), np.array([12, 13])]
    name = "stitch_example_6"
    input_dict = {"indices": indices, "data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger indices
    indices = [np.array([2, 5], dtype=np.int32), np.array([1, 4], dtype=np.int32)]
    data = [np.array([20, 23]), np.array([11, 22])]
    name = "stitch_example_7"
    input_dict = {"indices": indices, "data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Removing problematic input

    # Input 9: Removing problematic input

    # Input 10: More complex example with higher dimensions - Corrected Shape
    indices = [np.array([[0], [1]], dtype=np.int32), np.array([[2], [3]], dtype=np.int32)]
    data = [np.array([[[1, 2]], [[3, 4]]]), np.array([[[9, 10]], [[11, 12]]])]

    name = "stitch_example_10"
    input_dict = {"indices": indices, "data": data, "name": name}

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.dynamic_stitch"] = tf_dynamic_stitch_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.dynamic_stitch' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.dynamic_stitch'.")

check_valid('tf.dynamic_stitch', generated_inputs['tf.dynamic_stitch'], lib="tf", suffix=0)
