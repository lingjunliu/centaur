
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Cumprod_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([1, 2, 3], dtype=np.int32)
    axis = np.array(0, dtype=np.int32)
    exclusive = False
    reverse = False
    name = "cumprod_1"

    input_dict = {
        "x": x,
        "axis": axis,
        "exclusive": exclusive,
        "reverse": reverse,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    axis = np.array(0, dtype=np.int32)
    exclusive = True
    reverse = False
    name = "cumprod_2"

    input_dict = {
        "x": x,
        "axis": axis,
        "exclusive": exclusive,
        "reverse": reverse,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([1, 2, 3], dtype=np.int64)
    axis = np.array(0, dtype=np.int64)
    exclusive = False
    reverse = True
    name = "cumprod_3"

    input_dict = {
        "x": x,
        "axis": axis,
        "exclusive": exclusive,
        "reverse": reverse,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([[1, 2], [3, 4]], dtype=np.float64)
    axis = np.array(1, dtype=np.int32)
    exclusive = True
    reverse = True
    name = "cumprod_4"

    input_dict = {
        "x": x,
        "axis": axis,
        "exclusive": exclusive,
        "reverse": reverse,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([[-1, 2], [-3, 4]], dtype=np.int32)
    axis = np.array(0, dtype=np.int32)
    exclusive = False
    reverse = False
    name = "cumprod_5"

    input_dict = {
        "x": x,
        "axis": axis,
        "exclusive": exclusive,
        "reverse": reverse,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
    axis = np.array(1, dtype=np.int32)
    exclusive = True
    reverse = False
    name = "cumprod_6"

    input_dict = {
        "x": x,
        "axis": axis,
        "exclusive": exclusive,
        "reverse": reverse,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7
    x = np.array([1, 2, 3], dtype=np.uint8)
    axis = np.array(0, dtype=np.int32)
    exclusive = False
    reverse = False
    name = "cumprod_7"

    input_dict = {
        "x": x,
        "axis": axis,
        "exclusive": exclusive,
        "reverse": reverse,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex64)
    axis = np.array(0, dtype=np.int32)
    exclusive = False
    reverse = False
    name = "cumprod_8"

    input_dict = {
        "x": x,
        "axis": axis,
        "exclusive": exclusive,
        "reverse": reverse,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.array([1, 2, 3, 4, 5], dtype=np.int16)
    axis = np.array(0, dtype=np.int32)
    exclusive = False
    reverse = False
    name = "cumprod_9"

    input_dict = {
        "x": x,
        "axis": axis,
        "exclusive": exclusive,
        "reverse": reverse,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.array([1, 2, 3, 4, 5], dtype=np.int8)
    axis = np.array(0, dtype=np.int32)
    exclusive = False
    reverse = False
    name = "cumprod_10"

    input_dict = {
        "x": x,
        "axis": axis,
        "exclusive": exclusive,
        "reverse": reverse,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Cumprod"] = tf_raw_ops_Cumprod_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Cumprod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Cumprod'.")

check_valid('tf.raw_ops.Cumprod', generated_inputs['tf.raw_ops.Cumprod'], lib="tf", suffix=0)
