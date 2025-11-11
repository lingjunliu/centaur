
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_strided_slice_inputs():
    list_of_inputs = []
    input_1 = tf.constant(np.arange(10).reshape(2, 5), dtype=tf.int32)
    begin_1 = np.array([0, 0], dtype=np.int32)
    end_1 = np.array([2, 3], dtype=np.int32)
    strides_1 = np.array([1, 1], dtype=np.int32)
    begin_mask_1 = 0
    end_mask_1 = 0
    ellipsis_mask_1 = 0
    new_axis_mask_1 = 0
    shrink_axis_mask_1 = 0
    var_1 = None
    name_1 = "slice_1"
    input_dict_1 = {
        "input_": input_1,
        "begin": begin_1,
        "end": end_1,
        "strides": strides_1,
        "begin_mask": begin_mask_1,
        "end_mask": end_mask_1,
        "ellipsis_mask": ellipsis_mask_1,
        "new_axis_mask": new_axis_mask_1,
        "shrink_axis_mask": shrink_axis_mask_1,
        "var": var_1,
        "name": name_1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    input_2 = tf.constant(np.arange(24).reshape(2, 3, 4), dtype=tf.int32)
    begin_2 = np.array([0, 1, 2], dtype=np.int32)
    end_2 = np.array([2, 3, 4], dtype=np.int32)
    strides_2 = np.array([1, 1, 1], dtype=np.int32)
    begin_mask_2 = 0
    end_mask_2 = 0
    ellipsis_mask_2 = 0
    new_axis_mask_2 = 0
    shrink_axis_mask_2 = 0
    var_2 = None
    name_2 = "slice_2"
    input_dict_2 = {
        "input_": input_2,
        "begin": begin_2,
        "end": end_2,
        "strides": strides_2,
        "begin_mask": begin_mask_2,
        "end_mask": end_mask_2,
        "ellipsis_mask": ellipsis_mask_2,
        "new_axis_mask": new_axis_mask_2,
        "shrink_axis_mask": shrink_axis_mask_2,
        "var": var_2,
        "name": name_2
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    input_3 = tf.constant(np.arange(12).reshape(3, 4), dtype=tf.int32)
    begin_3 = np.array([0, 0], dtype=np.int32)
    end_3 = np.array([3, 4], dtype=np.int32)
    strides_3 = np.array([-1, 1], dtype=np.int32)
    begin_mask_3 = 0
    end_mask_3 = 0
    ellipsis_mask_3 = 0
    new_axis_mask_3 = 0
    shrink_axis_mask_3 = 0
    var_3 = None
    name_3 = "slice_3"
    input_dict_3 = {
        "input_": input_3,
        "begin": begin_3,
        "end": end_3,
        "strides": strides_3,
        "begin_mask": begin_mask_3,
        "end_mask": end_mask_3,
        "ellipsis_mask": ellipsis_mask_3,
        "new_axis_mask": new_axis_mask_3,
        "shrink_axis_mask": shrink_axis_mask_3,
        "var": var_3,
        "name": name_3
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    input_4 = tf.constant(np.arange(8).reshape(2, 2, 2), dtype=tf.int32)
    begin_4 = np.array([0, 0, 0], dtype=np.int32)
    end_4 = np.array([1, 2, 2], dtype=np.int32)
    strides_4 = np.array([1, 1, 1], dtype=np.int32)
    begin_mask_4 = 1
    end_mask_4 = 0
    ellipsis_mask_4 = 0
    new_axis_mask_4 = 0
    shrink_axis_mask_4 = 0
    var_4 = None
    name_4 = "slice_4"
    input_dict_4 = {
        "input_": input_4,
        "begin": begin_4,
        "end": end_4,
        "strides": strides_4,
        "begin_mask": begin_mask_4,
        "end_mask": end_mask_4,
        "ellipsis_mask": ellipsis_mask_4,
        "new_axis_mask": new_axis_mask_4,
        "shrink_axis_mask": shrink_axis_mask_4,
        "var": var_4,
        "name": name_4
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    input_5 = tf.constant(np.arange(16).reshape(4, 4), dtype=tf.int32)
    begin_5 = np.array([0, 0], dtype=np.int32)
    end_5 = np.array([4, 4], dtype=np.int32)
    strides_5 = np.array([1, 1], dtype=np.int32)
    begin_mask_5 = 0
    end_mask_5 = 1
    ellipsis_mask_5 = 0
    new_axis_mask_5 = 0
    shrink_axis_mask_5 = 0
    var_5 = None
    name_5 = "slice_5"
    input_dict_5 = {
        "input_": input_5,
        "begin": begin_5,
        "end": end_5,
        "strides": strides_5,
        "begin_mask": begin_mask_5,
        "end_mask": end_mask_5,
        "ellipsis_mask": ellipsis_mask_5,
        "new_axis_mask": new_axis_mask_5,
        "shrink_axis_mask": shrink_axis_mask_5,
        "var": var_5,
        "name": name_5
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    input_6 = tf.constant(np.arange(20).reshape(5, 4), dtype=tf.int32)
    begin_6 = np.array([0, 0], dtype=np.int32)
    end_6 = np.array([5, 4], dtype=np.int32)
    strides_6 = np.array([1, 2], dtype=np.int32)
    begin_mask_6 = 0
    end_mask_6 = 0
    ellipsis_mask_6 = 0
    new_axis_mask_6 = 0
    shrink_axis_mask_6 = 0
    var_6 = None
    name_6 = "slice_6"
    input_dict_6 = {
        "input_": input_6,
        "begin": begin_6,
        "end": end_6,
        "strides": strides_6,
        "begin_mask": begin_mask_6,
        "end_mask": end_mask_6,
        "ellipsis_mask": ellipsis_mask_6,
        "new_axis_mask": new_axis_mask_6,
        "shrink_axis_mask": shrink_axis_mask_6,
        "var": var_6,
        "name": name_6
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    input_7 = tf.constant(np.arange(10).reshape(5, 2), dtype=tf.int32)
    begin_7 = np.array([0, 0], dtype=np.int32)
    end_7 = np.array([5, 2], dtype=np.int32)
    strides_7 = np.array([2, 1], dtype=np.int32)
    begin_mask_7 = 0
    end_mask_7 = 0
    ellipsis_mask_7 = 0
    new_axis_mask_7 = 1
    shrink_axis_mask_7 = 0
    var_7 = None
    name_7 = "slice_7"
    input_dict_7 = {
        "input_": input_7,
        "begin": begin_7,
        "end": end_7,
        "strides": strides_7,
        "begin_mask": begin_mask_7,
        "end_mask": end_mask_7,
        "ellipsis_mask": ellipsis_mask_7,
        "new_axis_mask": new_axis_mask_7,
        "shrink_axis_mask": shrink_axis_mask_7,
        "var": var_7,
        "name": name_7
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    input_8 = tf.constant(np.arange(12).reshape(3, 4), dtype=tf.int32)
    begin_8 = np.array([0, 1], dtype=np.int32)
    end_8 = np.array([3, 3], dtype=np.int32)
    strides_8 = np.array([1, 1], dtype=np.int32)
    begin_mask_8 = 0
    end_mask_8 = 0
    ellipsis_mask_8 = 0
    new_axis_mask_8 = 0
    shrink_axis_mask_8 = 2
    var_8 = None
    name_8 = "slice_8"
    input_dict_8 = {
        "input_": input_8,
        "begin": begin_8,
        "end": end_8,
        "strides": strides_8,
        "begin_mask": begin_mask_8,
        "end_mask": end_mask_8,
        "ellipsis_mask": ellipsis_mask_8,
        "new_axis_mask": new_axis_mask_8,
        "shrink_axis_mask": shrink_axis_mask_8,
        "var": var_8,
        "name": name_8
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    input_9 = tf.constant(np.arange(24).reshape(2, 3, 4), dtype=tf.int32)
    begin_9 = np.array([0, 0, 0], dtype=np.int32)
    end_9 = np.array([1, 3, 4], dtype=np.int32)
    strides_9 = np.array([1, 1, 2], dtype=np.int32)
    begin_mask_9 = 0
    end_mask_9 = 0
    ellipsis_mask_9 = 0
    new_axis_mask_9 = 0
    shrink_axis_mask_9 = 0
    var_9 = None
    name_9 = "slice_9"
    input_dict_9 = {
        "input_": input_9,
        "begin": begin_9,
        "end": end_9,
        "strides": strides_9,
        "begin_mask": begin_mask_9,
        "end_mask": end_mask_9,
        "ellipsis_mask": ellipsis_mask_9,
        "new_axis_mask": new_axis_mask_9,
        "shrink_axis_mask": shrink_axis_mask_9,
        "var": var_9,
        "name": name_9
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    input_10 = tf.constant(np.arange(30).reshape(5, 3, 2), dtype=tf.int32)
    begin_10 = np.array([1, 0, 0], dtype=np.int32)
    end_10 = np.array([4, 3, 2], dtype=np.int32)
    strides_10 = np.array([1, 1, 1], dtype=np.int32)
    begin_mask_10 = 0
    end_mask_10 = 0
    ellipsis_mask_10 = 0
    new_axis_mask_10 = 0
    shrink_axis_mask_10 = 0
    var_10 = None
    name_10 = "slice_10"
    input_dict_10 = {
        "input_": input_10,
        "begin": begin_10,
        "end": end_10,
        "strides": strides_10,
        "begin_mask": begin_mask_10,
        "end_mask": end_mask_10,
        "ellipsis_mask": ellipsis_mask_10,
        "new_axis_mask": new_axis_mask_10,
        "shrink_axis_mask": shrink_axis_mask_10,
        "var": var_10,
        "name": name_10
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.strided_slice"] = tf_strided_slice_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.strided_slice' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strided_slice'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.strided_slice', generated_inputs['tf.strided_slice'], lib="tf", suffix=0)
