
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ClipByValue_inputs():
    list_of_inputs = []

    # Input 1: float32, scalar min/max
    t = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    clip_value_min = np.array(0.0, dtype=np.float32)
    clip_value_max = np.array(1.0, dtype=np.float32)
    input_dict = {"t": t, "clip_value_min": clip_value_min, "clip_value_max": clip_value_max, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int32, scalar min/max
    t = np.array([-1, 0, 1, 2], dtype=np.int32)
    clip_value_min = np.array(0, dtype=np.int32)
    clip_value_max = np.array(1, dtype=np.int32)
    input_dict = {"t": t, "clip_value_min": clip_value_min, "clip_value_max": clip_value_max, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64, array min/max
    t = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float64).reshape((2,2))
    clip_value_min = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    clip_value_max = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    input_dict = {"t": t, "clip_value_min": clip_value_min, "clip_value_max": clip_value_max, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int64, scalar min/max, negative min
    t = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
    clip_value_min = np.array(-1, dtype=np.int64)
    clip_value_max = np.array(1, dtype=np.int64)
    input_dict = {"t": t, "clip_value_min": clip_value_min, "clip_value_max": clip_value_max, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: uint8, scalar min/max
    t = np.array([0, 100, 200, 255], dtype=np.uint8)
    clip_value_min = np.array(50, dtype=np.uint8)
    clip_value_max = np.array(150, dtype=np.uint8)
    input_dict = {"t": t, "clip_value_min": clip_value_min, "clip_value_max": clip_value_max, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float16, scalar min/max
    t = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float16)
    clip_value_min = np.array(-1.0, dtype=np.float16)
    clip_value_max = np.array(1.0, dtype=np.float16)
    input_dict = {"t": t, "clip_value_min": clip_value_min, "clip_value_max": clip_value_max, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7: half, scalar min/max
    t = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float16)
    clip_value_min = np.array(0.0, dtype=np.float16)
    clip_value_max = np.array(1.0, dtype=np.float16)
    input_dict = {"t": t, "clip_value_min": clip_value_min, "clip_value_max": clip_value_max, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int16, scalar min/max
    t = np.array([-1, 0, 1, 2], dtype=np.int16)
    clip_value_min = np.array(0, dtype=np.int16)
    clip_value_max = np.array(1, dtype=np.int16)
    input_dict = {"t": t, "clip_value_min": clip_value_min, "clip_value_max": clip_value_max, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D tensor
    t = np.arange(24, dtype=np.float32).reshape((2, 3, 4))
    clip_value_min = np.array(5.0, dtype=np.float32)
    clip_value_max = np.array(15.0, dtype=np.float32)
    input_dict = {"t": t, "clip_value_min": clip_value_min, "clip_value_max": clip_value_max, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32, array min/max
    t = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32).reshape((2,2))
    clip_value_min = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    clip_value_max = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    input_dict = {"t": t, "clip_value_min": clip_value_min, "clip_value_max": clip_value_max, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.ClipByValue"] = tf_raw_ops_ClipByValue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ClipByValue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ClipByValue'.")

check_valid('tf.raw_ops.ClipByValue', generated_inputs['tf.raw_ops.ClipByValue'], lib="tf", suffix=0)
