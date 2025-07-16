
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_concat_inputs():
    list_of_inputs = []

    def create_input_dict(values, axis, name):
        return {"values": values, "axis": axis, "name": name}

    # Input 1: Basic 2D concatenation along axis 0
    values1 = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    input_dict1 = create_input_dict(values1, 0, "concat_2d_axis0")
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Basic 2D concatenation along axis 1
    values2 = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    input_dict2 = create_input_dict(values2, 1, "concat_2d_axis1")
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D concatenation along axis 0
    values3 = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])]
    input_dict3 = create_input_dict(values3, 0, "concat_3d_axis0")
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Concatenation with negative axis
    values4 = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    input_dict4 = create_input_dict(values4, -1, "concat_negative_axis")
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Concatenating 1D arrays
    values5 = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    input_dict5 = create_input_dict(values5, 0, "concat_1d_axis0")
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.concat"] = tf_concat_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.concat' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.concat'.")

check_valid('tf.concat', generated_inputs['tf.concat'], lib="tf", suffix=0)
