
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_fill_inputs():
    list_of_inputs = []

    # Input 1
    dims = [2, 3]
    value = np.int32(9)
    name = "fill_tensor_1"
    layout = None
    input_dict = {"dims": dims, "value": value, "name": name, "layout": layout}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    dims = [5]
    value = np.float32(3.14)
    name = "fill_tensor_2"
    layout = None
    input_dict = {"dims": dims, "value": value, "name": name, "layout": layout}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    dims = [1, 4, 2]
    value = np.int64(-1)
    name = "fill_tensor_3"
    layout = None
    input_dict = {"dims": dims, "value": value, "name": name, "layout": layout}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    dims = [2, 2, 2, 2]
    value = np.bool_(True)
    name = "fill_tensor_4"
    layout = None
    input_dict = {"dims": dims, "value": value, "name": name, "layout": layout}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    dims = [3, 1]
    value = np.array("hello").astype(np.object_) # Changed to object type
    name = "fill_tensor_5"
    layout = None
    input_dict = {"dims": dims, "value": value, "name": name, "layout": layout}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    dims = [1]
    value = np.int32(0)
    name = "fill_tensor_6"
    layout = None
    input_dict = {"dims": dims, "value": value, "name": name, "layout": layout}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    dims = [4, 4]
    value = np.float64(1.618)
    name = "fill_tensor_7"
    layout = None
    input_dict = {"dims": dims, "value": value, "name": name, "layout": layout}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    dims = [1, 1, 1, 1, 1]
    value = np.uint8(255)
    name = "fill_tensor_8"
    layout = None
    input_dict = {"dims": dims, "value": value, "name": name, "layout": layout}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    dims = [2]
    value = np.float16(-1.0)
    name = "fill_tensor_9"
    layout = None
    input_dict = {"dims": dims, "value": value, "name": name, "layout": layout}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    dims = [3, 2, 1]
    value = np.int16(1000)  # reduced to avoid overflow.
    name = "fill_tensor_10"
    layout = None
    input_dict = {"dims": dims, "value": value, "name": name, "layout": layout}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.fill"] = tf_fill_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.fill' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.fill'.")

check_valid('tf.fill', generated_inputs['tf.fill'], lib="tf", suffix=0)
