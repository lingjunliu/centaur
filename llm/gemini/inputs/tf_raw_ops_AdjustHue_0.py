
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_adjust_hue_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    images = np.random.rand(10, 10, 3).astype(np.float32)
    delta = np.array(0.5, dtype=np.float32)
    input_dict = {"images": images, "delta": delta, "name": "adjust_hue_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different image size
    images = np.random.rand(20, 30, 3).astype(np.float32)
    delta = np.array(-0.2, dtype=np.float32)
    input_dict = {"images": images, "delta": delta, "name": "adjust_hue_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Half precision images
    images = np.random.rand(5, 5, 3).astype(np.float16)
    delta = np.array(1.0, dtype=np.float32)
    input_dict = {"images": images, "delta": delta, "name": "adjust_hue_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D images
    images = np.random.rand(2, 10, 10, 3).astype(np.float32)
    delta = np.array(0.75, dtype=np.float32)
    input_dict = {"images": images, "delta": delta, "name": "adjust_hue_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large delta
    images = np.random.rand(8, 8, 3).astype(np.float32)
    delta = np.array(5.0, dtype=np.float32)
    input_dict = {"images": images, "delta": delta, "name": "adjust_hue_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative delta
    images = np.random.rand(12, 12, 3).astype(np.float32)
    delta = np.array(-2.5, dtype=np.float32)
    input_dict = {"images": images, "delta": delta, "name": "adjust_hue_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small delta
    images = np.random.rand(4, 4, 3).astype(np.float32)
    delta = np.array(0.01, dtype=np.float32)
    input_dict = {"images": images, "delta": delta, "name": "adjust_hue_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different dimensions
    images = np.random.rand(1, 5, 5, 3).astype(np.float32)
    delta = np.array(0.3, dtype=np.float32)
    input_dict = {"images": images, "delta": delta, "name": "adjust_hue_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Zero delta
    images = np.random.rand(7, 7, 3).astype(np.float32)
    delta = np.array(0.0, dtype=np.float32)
    input_dict = {"images": images, "delta": delta, "name": "adjust_hue_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Another image size and delta
    images = np.random.rand(15, 25, 3).astype(np.float32)
    delta = np.array(0.9, dtype=np.float32)
    input_dict = {"images": images, "delta": delta, "name": "adjust_hue_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.AdjustHue"] = tf_raw_ops_adjust_hue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.AdjustHue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AdjustHue'.")

check_valid('tf.raw_ops.AdjustHue', generated_inputs['tf.raw_ops.AdjustHue'], lib="tf", suffix=0)
