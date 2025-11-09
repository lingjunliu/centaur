
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_image_stateless_random_hue_inputs():
    list_of_inputs = []
    rs = np.random.RandomState(42)

    # Input 1
    image = np.array([[[0.1, 0.2, 0.3],
                       [0.4, 0.5, 0.6]],
                      [[0.7, 0.8, 0.9],
                       [0.2, 0.3, 0.4]]], dtype=np.float32)
    max_delta = 0.2
    seed = np.array([1, 2], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 2
    image = rs.rand(4, 4, 3).astype(np.float32)
    max_delta = 0.5
    seed = np.array([123456789, 987654321], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 3 (batched)
    image = rs.rand(3, 8, 8, 3).astype(np.float32)
    max_delta = 0.0
    seed = np.array([0, 0], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 4 (float16)
    image = np.array([[[0.1, 0.2, 0.3]]], dtype=np.float16)
    max_delta = 0.3
    seed = np.array([2025, 1108], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 5 (all ones)
    image = np.ones((10, 10, 3), dtype=np.float32)
    max_delta = 0.05
    seed = np.array([42, 24], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 6 (batched with H=1)
    image = rs.rand(2, 1, 5, 3).astype(np.float32)
    max_delta = 0.49
    seed = np.array([31415, 27182], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 7 (all zeros)
    image = np.zeros((224, 224, 3), dtype=np.float32)
    max_delta = 0.4
    seed = np.array([7, 11], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 8 (float32 small image)
    image = np.array([[[0.95, 0.05, 0.5],
                       [0.25, 0.75, 0.5],
                       [0.0, 1.0, 0.5]]], dtype=np.float32)
    max_delta = 0.1
    seed = np.array([1001, 2002], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 9 (batched, small)
    image = rs.rand(5, 3, 3, 3).astype(np.float32)
    max_delta = 0.3
    seed = np.array([123, 456], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 10 (non-square image)
    image = rs.rand(7, 5, 3).astype(np.float32)
    max_delta = 0.2
    seed = np.array([8080, 9090], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 11 (values near 0 and 1)
    image = np.vstack([
        np.zeros((8, 16, 3), dtype=np.float32),
        np.ones((8, 16, 3), dtype=np.float32)
    ])
    max_delta = 0.0001
    seed = np.array([314, 159], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 12 (batched 2 images 3x3x3)
    image = np.array([
        [[[0.2, 0.3, 0.4],
          [0.5, 0.6, 0.7],
          [0.8, 0.9, 1.0]],
         [[0.1, 0.2, 0.3],
          [0.4, 0.5, 0.6],
          [0.7, 0.8, 0.9]],
         [[0.9, 0.8, 0.7],
          [0.6, 0.5, 0.4],
          [0.3, 0.2, 0.1]]],
        [[[1.0, 0.9, 0.8],
          [0.7, 0.6, 0.5],
          [0.4, 0.3, 0.2]],
         [[0.2, 0.1, 0.0],
          [0.3, 0.4, 0.5],
          [0.6, 0.7, 0.8]],
         [[0.05, 0.95, 0.5],
          [0.25, 0.75, 0.5],
          [0.45, 0.55, 0.5]]]
    ], dtype=np.float32)
    max_delta = 0.25
    seed = np.array([999, 111], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    return list_of_inputs

generated_inputs["tf.image.stateless_random_hue_1"] = tf_image_stateless_random_hue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.stateless_random_hue_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.stateless_random_hue_1'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.stateless_random_hue', generated_inputs['tf.image.stateless_random_hue_1'], lib="tf", suffix=1)
