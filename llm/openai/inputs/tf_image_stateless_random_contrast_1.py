
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_stateless_random_contrast_inputs():
    list_of_inputs = []

    image = np.array([[[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]],
                      [[7.0, 8.0, 9.0],
                       [10.0, 11.0, 12.0]]], dtype=np.float64)
    lower = np.float64(0.2)
    upper = np.float64(0.5)
    seed = np.array([1, 2], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    image = (np.arange(1*3*3*1).reshape(1, 3, 3, 1)).astype(np.float64)
    lower = np.float64(0.5)
    upper = np.float64(1.5)
    seed = np.array([12345, 67890], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    image = (np.random.RandomState(0).randn(2, 1, 4, 4, 3)).astype(np.float64)
    lower = np.float64(0.1)
    upper = np.float64(0.9)
    seed = np.array([7, 11], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    image = (np.linspace(-5, 5, 4*4*3).reshape(4, 4, 3)).astype(np.float64)
    lower = np.float64(0.8)
    upper = np.float64(1.2)
    seed = np.array([42, 24], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    image = (np.random.RandomState(1).rand(1, 2, 2, 4) * 255.0).astype(np.float64)
    lower = np.float64(1.0)
    upper = np.float64(2.0)
    seed = np.array([0, 999], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    image = (np.random.RandomState(2).rand(5, 5, 1) * 100.0 - 50.0).astype(np.float64)
    lower = np.float64(0.2)
    upper = np.float64(0.3)
    seed = np.array([31415, 92653], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    image = (np.random.RandomState(3).randn(3, 8, 8, 3)).astype(np.float64)
    lower = np.float64(0.0)
    upper = np.float64(3.0)
    seed = np.array([13579, 24680], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    image = np.array([[[0.5]]], dtype=np.float64)
    lower = np.float64(0.9)
    upper = np.float64(1.1)
    seed = np.array([101, 202], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    image = (np.random.RandomState(4).rand(2, 1, 1, 2)).astype(np.float64)
    lower = np.float64(0.01)
    upper = np.float64(0.02)
    seed = np.array([1, 3], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    image = (np.random.RandomState(5).randn(1, 2, 3, 3, 3)).astype(np.float64)
    lower = np.float64(0.3)
    upper = np.float64(0.3001)
    seed = np.array([777, 888], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    return list_of_inputs

generated_inputs["tf.image.stateless_random_contrast_1"] = tf_image_stateless_random_contrast_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.stateless_random_contrast_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.stateless_random_contrast_1'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.stateless_random_contrast', generated_inputs['tf.image.stateless_random_contrast_1'], lib="tf", suffix=1)
