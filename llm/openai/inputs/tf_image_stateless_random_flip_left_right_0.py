
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import torch
import copy

def tf_image_stateless_random_flip_left_right_inputs():
    list_of_inputs = []

    # Input 1: 3D uint8 RGB image
    image = np.array(
        [
            [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
            [[100, 110, 120], [130, 140, 150], [160, 170, 180]],
        ],
        dtype=np.uint8,
    )
    seed = np.array([1, 2], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    # Input 2: 4D float32 grayscale batch
    image = np.array(
        [
            [[[0.1], [0.2], [0.3]],
             [[0.4], [0.5], [0.6]]],
            [[[1.0], [1.1], [1.2]],
             [[1.3], [1.4], [1.5]]],
        ],
        dtype=np.float32,
    )
    seed = np.array([123, 456], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    # Input 3: 3D float64 grayscale with negatives
    image = np.array(
        [
            [[-1.5], [2.0]],
            [[3.5], [-4.0]],
            [[5.25], [0.0]],
        ],
        dtype=np.float64,
    )
    seed = np.array([7, 0], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    # Input 4: 4D int32 with negative values
    image = np.array(
        [
            [
                [[-1, 2], [3, -4]],
                [[5, -6], [-7, 8]],
            ]
        ],
        dtype=np.int32,
    )
    seed = np.array([0, 0], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    # Input 5: 3D int32 width=1
    image = np.array(
        [
            [[-128]],
            [[-10]],
            [[10]],
            [[127]],
        ],
        dtype=np.int32,
    )
    seed = np.array([2021, 9], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    # Input 6: 3D int32 single row, 2 channels
    image = np.array(
        [
            [[1, -1], [2, -2], [3, -3], [4, -4]],
        ],
        dtype=np.int32,
    )
    seed = np.array([5, 6], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    # Input 7: 4D uint8 minimal dims
    image = np.array(
        [
            [[[0]]],
            [[[255]]],
        ],
        dtype=np.uint8,
    )  # shape (2,1,1,1)
    seed = np.array([8, 9], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    # Input 8: 4D float32 RGBA
    image = np.array(
        [
            [
                [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]],
                [[12.0, 13.0, 14.0, 15.0], [16.0, 17.0, 18.0, 19.0], [20.0, 21.0, 22.0, 23.0]],
            ],
            [
                [[0.5, 1.5, 2.5, 3.5], [4.5, 5.5, 6.5, 7.5], [8.5, 9.5, 10.5, 11.5]],
                [[12.5, 13.5, 14.5, 15.5], [16.5, 17.5, 18.5, 19.5], [20.5, 21.5, 22.5, 23.5]],
            ],
            [
                [[-1.0, -2.0, -3.0, -4.0], [-5.0, -6.0, -7.0, -8.0], [-9.0, -10.0, -11.0, -12.0]],
                [[-12.0, -13.0, -14.0, -15.0], [-16.0, -17.0, -18.0, -19.0], [-20.0, -21.0, -22.0, -23.0]],
            ],
        ],
        dtype=np.float32,
    )
    seed = np.array([31415, 27182], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    # Input 9: 3D float32 with NaN and Inf
    image = np.array(
        [
            [[np.nan, np.inf], [1.0, -1.0]],
            [[-np.inf, 0.0], [2.5, -3.5]],
        ],
        dtype=np.float32,
    )
    seed = np.array([100, 200], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    # Input 10: 4D float32 width=1, 3 channels
    image = np.array(
        [
            [
                [[0.0, 0.5, 1.0]],
                [[1.5, 2.0, 2.5]],
                [[-0.5, -1.0, -1.5]],
            ],
            [
                [[3.0, 3.5, 4.0]],
                [[4.5, 5.0, 5.5]],
                [[6.0, 6.5, 7.0]],
            ],
        ],
        dtype=np.float32,
    )
    seed = np.array([12, 34], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    # Input 11: 3D uint8 grayscale 5x5x1
    image = np.arange(25, dtype=np.uint8).reshape(5, 5, 1)
    seed = np.array([777, 888], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    # Input 12: 4D float32 1x4x4x3
    image = (np.arange(4 * 4 * 3, dtype=np.float32).reshape(1, 4, 4, 3) / np.float32(10.0))
    seed = np.array([65535, 1], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    return list_of_inputs

generated_inputs["tf.image.stateless_random_flip_left_right"] = tf_image_stateless_random_flip_left_right_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.stateless_random_flip_left_right' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.stateless_random_flip_left_right'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.stateless_random_flip_left_right', generated_inputs['tf.image.stateless_random_flip_left_right'], lib="tf", suffix=0)
