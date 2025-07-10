
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_signal_overlap_and_add_inputs():
    list_of_inputs = []

    # Input 1, valid
    signal = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.float32)
    frame_step = 1
    name = "overlap_add_1"
    input_dict = {"signal": signal, "frame_step": frame_step, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    signal = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    frame_step = 2
    name = "overlap_add_2"
    input_dict = {"signal": signal, "frame_step": frame_step, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    signal = np.array([[[1, 2], [3, 4], [5, 6]]], dtype=np.float32)
    frame_step = 1
    name = "overlap_add_3"
    input_dict = {"signal": signal, "frame_step": frame_step, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid
    signal = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]]], dtype=np.float32)
    frame_step = 3
    name = "overlap_add_4"
    input_dict = {"signal": signal, "frame_step": frame_step, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid
    signal = np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]], dtype=np.float32)
    frame_step = 2
    name = "overlap_add_5"
    input_dict = {"signal": signal, "frame_step": frame_step, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid, different data type
    signal = np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]], dtype=np.int32)
    frame_step = 2
    name = "overlap_add_6"
    input_dict = {"signal": signal, "frame_step": frame_step, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid, larger frame_step
    signal = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=np.float32)
    frame_step = 5
    name = "overlap_add_7"
    input_dict = {"signal": signal, "frame_step": frame_step, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid, more frames
    signal = np.array([[[1, 2], [3, 4], [5, 6], [7, 8]]], dtype=np.float32)
    frame_step = 1
    name = "overlap_add_8"
    input_dict = {"signal": signal, "frame_step": frame_step, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid
    signal = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=np.float32)
    frame_step = 3
    name = "overlap_add_9"
    input_dict = {"signal": signal, "frame_step": frame_step, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid, different shape
    signal = np.array([[[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]], dtype=np.float32)
    frame_step = 2
    name = "overlap_add_10"
    input_dict = {"signal": signal, "frame_step": frame_step, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.signal.overlap_and_add"] = tf_signal_overlap_and_add_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.signal.overlap_and_add' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.signal.overlap_and_add'.")

check_valid('tf.signal.overlap_and_add', generated_inputs['tf.signal.overlap_and_add'], lib="tf", suffix=0)
