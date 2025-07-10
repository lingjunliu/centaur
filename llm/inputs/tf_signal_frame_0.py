
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_signal_frame_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)
    frame_length = 3
    frame_step = 1
    pad_end = False
    pad_value = np.float32(0)
    axis = np.int32(-1)
    name = "frame1"
    input_dict = {"signal": signal, "frame_length": frame_length, "frame_step": frame_step, "pad_end": pad_end, "pad_value": pad_value, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: With padding
    signal = np.array([1, 2, 3, 4, 5]).astype(np.float32)
    frame_length = 3
    frame_step = 2
    pad_end = True
    pad_value = np.float32(0)
    axis = np.int32(-1)
    name = "frame2"
    input_dict = {"signal": signal, "frame_length": frame_length, "frame_step": frame_step, "pad_end": pad_end, "pad_value": pad_value, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multi-dimensional signal
    signal = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).astype(np.float32)
    frame_length = 2
    frame_step = 1
    pad_end = False
    pad_value = np.float32(0)
    axis = np.int32(1)
    name = "frame3"
    input_dict = {"signal": signal, "frame_length": frame_length, "frame_step": frame_step, "pad_end": pad_end, "pad_value": pad_value, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different pad value
    signal = np.array([1, 2, 3, 4, 5]).astype(np.float32)
    frame_length = 3
    frame_step = 2
    pad_end = True
    pad_value = np.float32(-1)
    axis = np.int32(-1)
    name = "frame4"
    input_dict = {"signal": signal, "frame_length": frame_length, "frame_step": frame_step, "pad_end": pad_end, "pad_value": pad_value, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger frame step
    signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)
    frame_length = 3
    frame_step = 4
    pad_end = False
    pad_value = np.float32(0)
    axis = np.int32(-1)
    name = "frame5"
    input_dict = {"signal": signal, "frame_length": frame_length, "frame_step": frame_step, "pad_end": pad_end, "pad_value": pad_value, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative axis
    signal = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).astype(np.float32)
    frame_length = 2
    frame_step = 1
    pad_end = False
    pad_value = np.float32(0)
    axis = np.int32(-1)
    name = "frame6"
    input_dict = {"signal": signal, "frame_length": frame_length, "frame_step": frame_step, "pad_end": pad_end, "pad_value": pad_value, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7: 3D signal
    signal = np.random.rand(2, 3, 4).astype(np.float32)
    frame_length = 2
    frame_step = 1
    pad_end = True
    pad_value = np.float32(0)
    axis = np.int32(2)
    name = "frame7"
    input_dict = {"signal": signal, "frame_length": frame_length, "frame_step": frame_step, "pad_end": pad_end, "pad_value": pad_value, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: frame_length > signal length, pad_end=False
    signal = np.array([1, 2, 3]).astype(np.float32)
    frame_length = 4
    frame_step = 1
    pad_end = False
    pad_value = np.float32(0)
    axis = np.int32(-1)
    name = "frame8"
    input_dict = {"signal": signal, "frame_length": frame_length, "frame_step": frame_step, "pad_end": pad_end, "pad_value": pad_value, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: frame_length > signal length, pad_end=True
    signal = np.array([1, 2, 3]).astype(np.float32)
    frame_length = 4
    frame_step = 1
    pad_end = True
    pad_value = np.float32(0)
    axis = np.int32(-1)
    name = "frame9"
    input_dict = {"signal": signal, "frame_length": frame_length, "frame_step": frame_step, "pad_end": pad_end, "pad_value": pad_value, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large frame_step, with padding.
    signal = np.array([1, 2, 3, 4, 5, 6]).astype(np.float32)
    frame_length = 2
    frame_step = 5
    pad_end = True
    pad_value = np.float32(0)
    axis = np.int32(-1)
    name = "frame10"
    input_dict = {"signal": signal, "frame_length": frame_length, "frame_step": frame_step, "pad_end": pad_end, "pad_value": pad_value, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Frame step equals Frame length.
    signal = np.array([1, 2, 3, 4, 5, 6]).astype(np.float32)
    frame_length = 2
    frame_step = 2
    pad_end = False
    pad_value = np.float32(0)
    axis = np.int32(-1)
    name = "frame11"
    input_dict = {"signal": signal, "frame_length": frame_length, "frame_step": frame_step, "pad_end": pad_end, "pad_value": pad_value, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.signal.frame"] = tf_signal_frame_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.signal.frame' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.signal.frame'.")

check_valid('tf.signal.frame', generated_inputs['tf.signal.frame'], lib="tf", suffix=0)
