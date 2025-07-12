
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_adjust_saturation_inputs():
    list_of_inputs = []

    # Input 1: Simple 3D float32 image, scale=1.0
    images = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]], dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    input_dict = {"images": images, "scale": scale, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Simple 3D float32 image, scale=0.5
    images = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]], dtype=np.float32)
    scale = np.array(0.5, dtype=np.float32)
    input_dict = {"images": images, "scale": scale, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Simple 3D float32 image, scale=2.0
    images = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]], dtype=np.float32)
    scale = np.array(2.0, dtype=np.float32)
    input_dict = {"images": images, "scale": scale, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D float32 image, scale=1.0
    images = np.random.rand(2, 2, 2, 3).astype(np.float32)
    scale = np.array(1.0, dtype=np.float32)
    input_dict = {"images": images, "scale": scale, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D half image, scale=0.75
    images = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]], dtype=np.float16)
    scale = np.array(0.75, dtype=np.float32)
    input_dict = {"images": images, "scale": scale, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D half image, scale=1.5
    images = np.random.rand(2, 3, 4, 3).astype(np.float16)
    scale = np.array(1.5, dtype=np.float32)
    input_dict = {"images": images, "scale": scale, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float32 image, scale=0.0
    images = np.array([[[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]], [[0.8, 0.9, 1.0], [1.1, 1.2, 1.3]]], dtype=np.float32)
    scale = np.array(0.0, dtype=np.float32)
    input_dict = {"images": images, "scale": scale, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D float32 image, scale=3.0
    images = np.array([[[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]], [[0.8, 0.9, 1.0], [1.1, 1.2, 1.3]]], dtype=np.float32)
    scale = np.array(3.0, dtype=np.float32)
    input_dict = {"images": images, "scale": scale, "name": "SaturationAdjust"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D float32 image, scale=1.0
    images = np.random.rand(1, 2, 2, 2, 3).astype(np.float32)
    scale = np.array(1.0, dtype=np.float32)
    input_dict = {"images": images, "scale": scale, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D half image with values between 0 and 1, scale=0.5
    images = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [0.0, 0.1, 0.2]]], dtype=np.float16)
    scale = np.array(0.5, dtype=np.float32)
    input_dict = {"images": images, "scale": scale, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.AdjustSaturation"] = []
inputs = tf_raw_ops_adjust_saturation_inputs()
for input_dict in inputs:
    generated_inputs["tf.raw_ops.AdjustSaturation"].append({
        "images": tf.convert_to_tensor(input_dict["images"]),
        "scale": tf.convert_to_tensor(input_dict["scale"]),
        "name": input_dict["name"]
    })

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.AdjustSaturation' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AdjustSaturation'.")

check_valid('tf.raw_ops.AdjustSaturation', generated_inputs['tf.raw_ops.AdjustSaturation'], lib="tf", suffix=0)
