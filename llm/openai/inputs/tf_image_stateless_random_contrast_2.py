
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_stateless_random_contrast_inputs():
    list_of_inputs = []

    # Input 1
    image = np.array([[[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]],
                      [[7.0, 8.0, 9.0],
                       [10.0, 11.0, 12.0]]], dtype=np.float32)
    lower = np.float32(0.2)
    upper = np.float32(0.5)
    seed = (np.int32(1), np.int32(2))
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    # Input 2
    image = (np.random.RandomState(0).randn(64, 64, 3).astype(np.float32) * 2.0) - 1.0
    lower = np.float32(0.8)
    upper = np.float32(1.2)
    seed = (np.int32(123), np.int32(456))
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    # Input 3
    image = np.random.RandomState(1).rand(1, 4, 4, 1).astype(np.float64)
    lower = np.float64(0.1)
    upper = np.float64(2.0)
    seed = (np.int32(0), np.int32(0))
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    # Input 4
    image = np.random.RandomState(2).randn(2, 3, 3, 3).astype(np.float32)
    lower = np.float32(0.5)
    upper = np.float32(1.5)
    seed = (np.int32(42), np.int32(24))
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    # Input 5
    image = (np.random.RandomState(3).randint(0, 256, size=(5, 5, 1))).astype(np.uint8)
    lower = np.float32(0.0)
    upper = np.float32(0.9)
    seed = (np.int32(7), np.int32(999999))
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    # Input 6
    image = np.random.RandomState(4).rand(2, 2, 2, 2, 3).astype(np.float32)
    lower = np.float32(1.0)
    upper = np.float32(1.1)
    seed = (np.int32(31415), np.int32(27182))
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    # Input 7
    image = np.random.RandomState(5).rand(10, 10, 4).astype(np.float32)
    lower = np.float32(0.3)
    upper = np.float32(0.7)
    seed = (np.int32(2021), np.int32(2022))
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    # Input 8
    image = np.ones((3, 2, 2, 3), dtype=np.float32)
    lower = np.float32(2.0)
    upper = np.float32(3.0)
    seed = (np.int32(111), np.int32(222))
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    # Input 9
    image = np.arange(1 * 1 * 3 * 3 * 1, dtype=np.float32).reshape((1, 1, 3, 3, 1))
    lower = np.float32(0.05)
    upper = np.float32(0.95)
    seed = (np.int32(8), np.int32(16))
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    # Input 10
    image = (np.random.RandomState(6).rand(128, 128, 3).astype(np.float32) * 1000.0)
    lower = np.float32(0.0001)
    upper = np.float32(0.01)
    seed = (np.int32(13579), np.int32(24680))
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    # Input 11
    image = np.random.RandomState(7).rand(2, 8, 8, 3).astype(np.float32)
    lower = np.float32(10.0)
    upper = np.float32(10.5)
    seed = (np.int32(77), np.int32(88))
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    # Input 12
    image = np.arange(1, 1 + 1 * 1 * 3, dtype=np.float64).reshape((1, 1, 3))
    lower = np.float64(0.25)
    upper = np.float64(0.75)
    seed = (np.int32(999), np.int32(1001))
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    return list_of_inputs

generated_inputs["tf.image.stateless_random_contrast_2"] = tf_image_stateless_random_contrast_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.stateless_random_contrast_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.stateless_random_contrast_2'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.stateless_random_contrast', generated_inputs['tf.image.stateless_random_contrast_2'], lib="tf", suffix=2)
