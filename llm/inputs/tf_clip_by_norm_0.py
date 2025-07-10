
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_clip_by_norm_inputs():
    list_of_inputs = []

    # Input 1
    t = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    clip_norm = np.array(2.0, dtype=np.float32)
    axes = None
    name = "clip_norm_example_1"
    input_dict = {"t": t, "clip_norm": clip_norm, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    t = np.array([[-1.0, -2.0], [3.0, 4.0]], dtype=np.float64)
    clip_norm = np.array(1.5, dtype=np.float64)
    axes = None
    name = "clip_norm_example_2"
    input_dict = {"t": t, "clip_norm": clip_norm, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    t = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    clip_norm = np.array(3.0, dtype=np.float32)
    axes = None
    name = "clip_norm_example_3"
    input_dict = {"t": t, "clip_norm": clip_norm, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    t = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    clip_norm = np.array(5.0, dtype=np.float32)
    axes = np.array([0], dtype=np.int32)
    name = "clip_norm_example_4"
    input_dict = {"t": t, "clip_norm": clip_norm, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    t = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    clip_norm = np.array(4.0, dtype=np.float32)
    axes = np.array([1], dtype=np.int32)
    name = "clip_norm_example_5"
    input_dict = {"t": t, "clip_norm": clip_norm, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    t = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    clip_norm = np.array(1.0, dtype=np.float64)
    axes = None
    name = "clip_norm_example_6"
    input_dict = {"t": t, "clip_norm": clip_norm, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    t = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    clip_norm = np.array(0.5, dtype=np.float32)
    axes = None
    name = "clip_norm_example_7"
    input_dict = {"t": t, "clip_norm": clip_norm, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    t = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    clip_norm = np.array(10.0, dtype=np.float32)
    axes = np.array([1,2], dtype=np.int32)
    name = "clip_norm_example_8"
    input_dict = {"t": t, "clip_norm": clip_norm, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 (Removed complex number example due to error)

    # Input 10 (Removed complex number example due to error)


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.clip_by_norm"] = tf_clip_by_norm_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.clip_by_norm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.clip_by_norm'.")

check_valid('tf.clip_by_norm', generated_inputs['tf.clip_by_norm'], lib="tf", suffix=0)
