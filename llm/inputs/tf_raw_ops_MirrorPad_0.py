
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_MirrorPad_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    paddings = np.array([[1, 1], [2, 2]], dtype=np.int32)
    mode = "SYMMETRIC"
    name = "mirror_pad_example_1"

    input_dict = {
        "input": input_tensor,
        "paddings": paddings,
        "mode": mode,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([1, 2, 3], dtype=np.int32)
    paddings = np.array([[0, 2]], dtype=np.int32)
    mode = "REFLECT"
    name = "mirror_pad_example_2"

    input_dict = {
        "input": input_tensor,
        "paddings": paddings,
        "mode": mode,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    paddings = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.int32)
    mode = "SYMMETRIC"
    name = "mirror_pad_example_3"

    input_dict = {
        "input": input_tensor,
        "paddings": paddings,
        "mode": mode,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.int64)
    paddings = np.array([[0, 0], [0, 0]], dtype=np.int64)
    mode = "REFLECT"
    name = "mirror_pad_example_4"

    input_dict = {
        "input": input_tensor,
        "paddings": paddings,
        "mode": mode,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([1], dtype=np.int32)
    paddings = np.array([[0, 0]], dtype=np.int32)
    mode = "SYMMETRIC"
    name = "mirror_pad_example_5"

    input_dict = {
        "input": input_tensor,
        "paddings": paddings,
        "mode": mode,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    paddings = np.array([[0, 0], [0, 0]], dtype=np.int32)
    mode = "REFLECT"
    name = "mirror_pad_example_6"

    input_dict = {
        "input": input_tensor,
        "paddings": paddings,
        "mode": mode,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    paddings = np.array([[1, 0], [0, 1]], dtype=np.int32)
    mode = "SYMMETRIC"
    name = "mirror_pad_example_7"

    input_dict = {
        "input": input_tensor,
        "paddings": paddings,
        "mode": mode,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    paddings = np.array([[0, 0], [0, 0], [0, 0]], dtype=np.int32)
    mode = "REFLECT"
    name = "mirror_pad_example_8"

    input_dict = {
        "input": input_tensor,
        "paddings": paddings,
        "mode": mode,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    paddings = np.array([[1, 0]], dtype=np.int32)
    mode = "SYMMETRIC"
    name = "mirror_pad_example_9"

    input_dict = {
        "input": input_tensor,
        "paddings": paddings,
        "mode": mode,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.int64)
    paddings = np.array([[1, 1], [1, 1]], dtype=np.int64)
    mode = "REFLECT"
    name = "mirror_pad_example_10"

    input_dict = {
        "input": input_tensor,
        "paddings": paddings,
        "mode": mode,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs["tf.raw_ops.MirrorPad"] = tf_raw_ops_MirrorPad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MirrorPad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MirrorPad'.")

check_valid('tf.raw_ops.MirrorPad', generated_inputs['tf.raw_ops.MirrorPad'], lib="tf", suffix=0)
