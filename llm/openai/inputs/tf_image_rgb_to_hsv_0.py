
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_rgb_to_hsv_inputs():
    list_of_inputs = []

    images = np.array([0.2, 0.4, 0.6], dtype=np.float32)
    name = "case1"
    list_of_inputs.append(copy.deepcopy({"images": images, "name": name}))

    images = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [-0.5, 2.0, 0.5],
        [0.3, -1.2, 1.5]
    ], dtype=np.float64)
    name = "case2"
    list_of_inputs.append(copy.deepcopy({"images": images, "name": name}))

    rng = np.random.RandomState(0)
    images = rng.rand(5, 5, 3).astype(np.float16)
    name = "case3"
    list_of_inputs.append(copy.deepcopy({"images": images, "name": name}))

    images = np.linspace(0, 1, 2 * 4 * 4 * 3, dtype=np.float32).reshape(2, 4, 4, 3)
    name = "case4"
    list_of_inputs.append(copy.deepcopy({"images": images, "name": name}))

    rng = np.random.RandomState(1)
    images = rng.randn(2, 1, 3, 4, 3).astype(np.float32)
    name = "case5"
    list_of_inputs.append(copy.deepcopy({"images": images, "name": name}))

    images = np.array([[1.0, 0.0, 0.0]], dtype=np.float16)
    name = "case6"
    list_of_inputs.append(copy.deepcopy({"images": images, "name": name}))

    images = np.tile(np.array([[[1.0, 0.0, 0.0]]], dtype=np.float64), (3, 3, 1))
    name = "case7"
    list_of_inputs.append(copy.deepcopy({"images": images, "name": name}))

    rng = np.random.RandomState(2)
    images = (rng.rand(10, 3).astype(np.float16) * 2.0) - 0.5
    name = "case8"
    list_of_inputs.append(copy.deepcopy({"images": images, "name": name}))

    images = np.mod(np.arange(7 * 1 * 3).reshape(7, 1, 3), 2).astype(np.float32)
    name = "case9"
    list_of_inputs.append(copy.deepcopy({"images": images, "name": name}))

    rng = np.random.RandomState(3)
    images = rng.rand(3, 2, 2, 3).astype(np.float32)
    name = "case10"
    list_of_inputs.append(copy.deepcopy({"images": images, "name": name}))

    return list_of_inputs

generated_inputs["tf.image.rgb_to_hsv"] = tf_image_rgb_to_hsv_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.rgb_to_hsv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.rgb_to_hsv'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.rgb_to_hsv', generated_inputs['tf.image.rgb_to_hsv'], lib="tf", suffix=0)
