
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strided_slice_inputs():
    list_of_inputs = []

    # Input 1
    input_ = np.array([[[1, 1, 1], [2, 2, 2]],
                       [[3, 3, 3], [4, 4, 4]],
                       [[5, 5, 5], [6, 6, 6]]], dtype=np.int32)
    begin = np.array([1, 0, 0], dtype=np.int32)
    end = np.array([2, 1, 3], dtype=np.int32)
    strides = np.array([1, 1, 1], dtype=np.int32)
    begin_mask = 0
    end_mask = 0
    ellipsis_mask = 0
    new_axis_mask = 0
    shrink_axis_mask = 0
    var = None
    name = "slice1"

    input_dict = {
        "input_": input_,
        "begin": begin,
        "end": end,
        "strides": strides,
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "ellipsis_mask": ellipsis_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask,
        "var": var,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_ = np.arange(24).reshape((2, 3, 4)).astype(np.int32)
    begin = np.array([0, 1, 1], dtype=np.int32)
    end = np.array([2, 2, 3], dtype=np.int32)
    strides = np.array([1, 1, 1], dtype=np.int32)
    begin_mask = 0
    end_mask = 0
    ellipsis_mask = 0
    new_axis_mask = 0
    shrink_axis_mask = 0
    var = None
    name = "slice2"

    input_dict = {
        "input_": input_,
        "begin": begin,
        "end": end,
        "strides": strides,
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "ellipsis_mask": ellipsis_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask,
        "var": var,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_ = np.arange(10).astype(np.int32)
    begin = np.array([2], dtype=np.int32)
    end = np.array([7], dtype=np.int32)
    strides = np.array([2], dtype=np.int32)
    begin_mask = 0
    end_mask = 0
    ellipsis_mask = 0
    new_axis_mask = 0
    shrink_axis_mask = 0
    var = None
    name = "slice3"

    input_dict = {
        "input_": input_,
        "begin": begin,
        "end": end,
        "strides": strides,
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "ellipsis_mask": ellipsis_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask,
        "var": var,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_ = np.arange(20).reshape((4, 5)).astype(np.int32)
    begin = np.array([1, 0], dtype=np.int32)
    end = np.array([3, 5], dtype=np.int32)
    strides = np.array([1, 1], dtype=np.int32)
    begin_mask = 0
    end_mask = 0
    ellipsis_mask = 0
    new_axis_mask = 0
    shrink_axis_mask = 0
    var = None
    name = "slice4"

    input_dict = {
        "input_": input_,
        "begin": begin,
        "end": end,
        "strides": strides,
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "ellipsis_mask": ellipsis_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask,
        "var": var,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_ = np.arange(12).reshape((2, 2, 3)).astype(np.int32)
    begin = np.array([0, 0, 0], dtype=np.int32)
    end = np.array([2, 2, 3], dtype=np.int32)
    strides = np.array([1, 1, 2], dtype=np.int32)
    begin_mask = 0
    end_mask = 0
    ellipsis_mask = 0
    new_axis_mask = 0
    shrink_axis_mask = 0
    var = None
    name = "slice5"

    input_dict = {
        "input_": input_,
        "begin": begin,
        "end": end,
        "strides": strides,
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "ellipsis_mask": ellipsis_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask,
        "var": var,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_ = np.arange(16).reshape((4, 4)).astype(np.int32)
    begin = np.array([1, 1], dtype=np.int32)
    end = np.array([3, 3], dtype=np.int32)
    strides = np.array([1, 1], dtype=np.int32)
    begin_mask = 0
    end_mask = 0
    ellipsis_mask = 0
    new_axis_mask = 0
    shrink_axis_mask = 0
    var = None
    name = "slice6"

    input_dict = {
        "input_": input_,
        "begin": begin,
        "end": end,
        "strides": strides,
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "ellipsis_mask": ellipsis_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask,
        "var": var,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 - Negative strides
    input_ = np.arange(10).astype(np.int32)
    begin = np.array([5], dtype=np.int32)
    end = np.array([2], dtype=np.int32)
    strides = np.array([-1], dtype=np.int32)
    begin_mask = 0
    end_mask = 0
    ellipsis_mask = 0
    new_axis_mask = 0
    shrink_axis_mask = 0
    var = None
    name = "slice7"

    input_dict = {
        "input_": input_,
        "begin": begin,
        "end": end,
        "strides": strides,
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "ellipsis_mask": ellipsis_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask,
        "var": var,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 - begin_mask and end_mask
    input_ = np.arange(20).reshape((4, 5)).astype(np.int32)
    begin = np.array([1, 0], dtype=np.int32)
    end = np.array([3, 3], dtype=np.int32)
    strides = np.array([1, 1], dtype=np.int32)
    begin_mask = 1
    end_mask = 2
    ellipsis_mask = 0
    new_axis_mask = 0
    shrink_axis_mask = 0
    var = None
    name = "slice8"

    input_dict = {
        "input_": input_,
        "begin": begin,
        "end": end,
        "strides": strides,
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "ellipsis_mask": ellipsis_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask,
        "var": var,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 - new_axis_mask
    input_ = np.arange(10).astype(np.int32)
    begin = np.array([2], dtype=np.int32)
    end = np.array([7], dtype=np.int32)
    strides = np.array([1], dtype=np.int32)
    begin_mask = 0
    end_mask = 0
    ellipsis_mask = 0
    new_axis_mask = 1
    shrink_axis_mask = 0
    var = None
    name = "slice9"

    input_dict = {
        "input_": input_,
        "begin": begin,
        "end": end,
        "strides": strides,
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "ellipsis_mask": ellipsis_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask,
        "var": var,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10 - shrink_axis_mask
    input_ = np.arange(20).reshape((4, 5)).astype(np.int32)
    begin = np.array([1, 0], dtype=np.int32)
    end = np.array([3, 5], dtype=np.int32)
    strides = np.array([1, 1], dtype=np.int32)
    begin_mask = 0
    end_mask = 0
    ellipsis_mask = 0
    new_axis_mask = 0
    shrink_axis_mask = 2
    var = None
    name = "slice10"

    input_dict = {
        "input_": input_,
        "begin": begin,
        "end": end,
        "strides": strides,
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "ellipsis_mask": ellipsis_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask,
        "var": var,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.strided_slice"] = tf_strided_slice_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.strided_slice' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strided_slice'.")

check_valid('tf.strided_slice', generated_inputs['tf.strided_slice'], lib="tf", suffix=0)
