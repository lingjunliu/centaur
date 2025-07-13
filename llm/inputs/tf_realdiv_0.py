
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_realdiv_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 division
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Division with negative values, float64
    x = np.array([-1.0, -2.0, 3.0], dtype=np.float64)
    y = np.array([0.5, -1.0, -1.5], dtype=np.float64)
    input_dict = {"x": x, "y": y, "name": "negative_division"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multi-dimensional array division, float32
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting division, float32
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([0.5, 1.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Integer division (will result in float), float32
    x = np.array([1, 2, 3], dtype=np.float32)
    y = np.array([2, 2, 2], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "integer_division"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large integer division, float64
    x = np.array([10000000000, 20000000000], dtype=np.float64)
    y = np.array([2, 5], dtype=np.float64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Division with zeros (will result in inf), float32
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "division_by_zero"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Division with uint8, float32
    x = np.array([1, 2, 3], dtype=np.float32)
    y = np.array([1, 2, 1], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Division with float16
    x = np.array([1.0, 2.0], dtype=np.float16)
    y = np.array([0.5, 1.0], dtype=np.float16)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Division with complex64
    x = np.array([1 + 1j, 2 + 2j], dtype=np.complex64)
    y = np.array([0.5 + 0.5j, 1 + 0j], dtype=np.complex64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.realdiv"] = tf_realdiv_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.realdiv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.realdiv'.")

check_valid('tf.realdiv', generated_inputs['tf.realdiv'], lib="tf", suffix=0)
