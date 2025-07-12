
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_adjust_hue_inputs():
    list_of_inputs = []

    # Input 1: Basic 3D float32 image and delta
    images = np.random.rand(10, 10, 3).astype(np.float32)
    delta = np.array(0.5, dtype=np.float32)
    input_dict = {"images": images, "delta": delta, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 4D float32 image and negative delta
    images = np.random.rand(1, 20, 20, 3).astype(np.float32)
    delta = np.array(-0.3, dtype=np.float32)
    input_dict = {"images": images, "delta": delta, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D half image and large delta
    images = np.random.rand(5, 5, 3).astype(np.float16)
    delta = np.array(2.0, dtype=np.float32)
    input_dict = {"images": images, "delta": delta, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D half image and small delta
    images = np.random.rand(2, 10, 10, 3).astype(np.float16)
    delta = np.array(0.1, dtype=np.float32)
    input_dict = {"images": images, "delta": delta, "name": "adjust_hue_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32 image with all zeros and delta.
    images = np.zeros((8, 8, 3), dtype=np.float32)
    delta = np.array(0.7, dtype=np.float32)
    input_dict = {"images": images, "delta": delta, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D half image and delta close to 1
    images = np.random.rand(7, 7, 3).astype(np.float16)
    delta = np.array(0.95, dtype=np.float32)
    input_dict = {"images": images, "delta": delta, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D float32 image with only one channel and delta.
    images = np.random.rand(1, 5, 5, 3).astype(np.float32)
    delta = np.array(-0.8, dtype=np.float32)
    input_dict = {"images": images, "delta": delta, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D half with a name
    images = np.random.rand(3, 4, 3).astype(np.float16)
    delta = np.array(0.6, dtype=np.float32)
    input_dict = {"images": images, "delta": delta, "name": "my_hue_adjust"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D float32, zero delta
    images = np.random.rand(6, 6, 3).astype(np.float32)
    delta = np.array(0.0, dtype=np.float32)
    input_dict = {"images": images, "delta": delta, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D half, delta greater than 1
    images = np.random.rand(2, 7, 7, 3).astype(np.float16)
    delta = np.array(1.5, dtype=np.float32)
    input_dict = {"images": images, "delta": delta, "name": None}
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
