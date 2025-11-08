
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_cumprod_inputs():
    list_of_inputs = []

    x = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    axis = 0
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

    x = np.array([1.0, 2.5, 3.2, 4.1], dtype=np.float64)
    axis = 0
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

    x = np.array([[1, 2], [3, 4]], dtype=np.int64)
    axis = 1
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

    x = np.array([[-1, 2], [-3, 4]], dtype=np.int32)
    axis = 0
    exclusive = True
    reverse = False
    name = "cumprod_4"
    input_dict = {
        "x": x,
        "axis": axis,
        "exclusive": exclusive,
        "reverse": reverse,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1, 2, 3], dtype=np.uint8)
    axis = 0
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

    x = np.array([1j, 2j, 3j], dtype=np.complex64)
    axis = 0
    exclusive = True
    reverse = True
    name = "cumprod_6"
    input_dict = {
        "x": x,
        "axis": axis,
        "exclusive": exclusive,
        "reverse": reverse,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    axis = 2
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

    x = np.array([1, 2, 3, 4], dtype=np.int16)
    axis = 0
    exclusive = False
    reverse = True
    name = "cumprod_8"
    input_dict = {
        "x": x,
        "axis": axis,
        "exclusive": exclusive,
        "reverse": reverse,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    axis = 0
    exclusive = True
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

    x = np.array([1, 2, 3], dtype=np.int8)
    axis = 0
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

generated_inputs["tf.math.cumprod"] = tf_math_cumprod_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.cumprod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.cumprod'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.cumprod', generated_inputs['tf.math.cumprod'], lib="tf", suffix=0)
