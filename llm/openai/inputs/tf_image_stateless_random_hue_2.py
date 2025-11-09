
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_stateless_random_hue_inputs():
    list_of_inputs = []

    # Input 1
    image = np.array([[[0.1, 0.2, 0.3],
                       [0.4, 0.5, 0.6]],
                      [[0.7, 0.8, 0.9],
                       [1.0, 1.1, 1.2]]], dtype=np.float32)
    max_delta = np.float32(0.2)
    seed = (np.int32(1), np.int32(2))
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 2
    image = (np.random.rand(64, 64, 3) * 255.0).astype(np.float32)
    max_delta = np.float64(0.5)
    seed = (np.int64(12345), np.int64(67890))
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 3
    image = (np.random.rand(8, 32, 32, 3)).astype(np.float16)
    max_delta = np.float32(0.0)
    seed = (np.int32(0), np.int32(0))
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 4
    image = np.array([[[0.2, -0.3, 1.5]]], dtype=np.float64)
    max_delta = np.float64(0.1)
    seed = (np.int64(-7), np.int64(42))
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 5
    image = (np.random.rand(10, 20, 3) * 2.0 - 1.0).astype(np.float32)
    max_delta = np.float32(0.05)
    seed = (np.int32(9999), np.int32(1))
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 6
    image = np.ones((1, 128, 128, 3), dtype=np.float32)
    max_delta = np.float32(0.3)
    seed = (np.int64(2021), np.int64(2022))
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 7
    image = np.arange(27, dtype=np.float16).reshape(3, 3, 3)
    max_delta = np.float32(0.49)
    seed = (np.int32(2147483647), np.int32(-2147483648))
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 8
    image = (np.random.rand(4, 5, 5, 3)).astype(np.float64)
    max_delta = np.float32(0.25)
    seed = (np.int64(0), np.int64(1))
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 9
    image = np.zeros((3, 4, 3), dtype=np.float32)
    max_delta = np.float32(0.001)
    seed = (np.int32(42), np.int32(24))
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 10
    image = np.array([[[1.0, 0.0, 0.5]],
                      [[0.3, 0.7, 0.2]]], dtype=np.float32)
    max_delta = np.float32(0.15)
    seed = (np.int64(555), np.int64(777))
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 11
    image = (np.random.rand(16, 16, 3) * 1000.0).astype(np.float16)
    max_delta = np.float32(0.33)
    seed = (np.int32(314159), np.int32(265358))
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 12
    image = (np.random.randn(2, 2, 2, 3)).astype(np.float64)
    max_delta = np.float64(0.5)
    seed = (np.int64(123), np.int64(456))
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    return list_of_inputs

generated_inputs["tf.image.stateless_random_hue_2"] = tf_image_stateless_random_hue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.stateless_random_hue_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.stateless_random_hue_2'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.stateless_random_hue', generated_inputs['tf.image.stateless_random_hue_2'], lib="tf", suffix=2)
