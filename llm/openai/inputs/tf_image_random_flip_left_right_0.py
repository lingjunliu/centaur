
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_random_flip_left_right_inputs():
    list_of_inputs = []

    image = np.array(
        [[[10, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120]],
         [[130, 140, 150], [160, 170, 180], [190, 200, 210], [220, 230, 240]],
         [[250, 0, 10], [20, 30, 40], [50, 60, 70], [80, 90, 100]]],
        dtype=np.uint8
    )
    seed = 1
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.array(
        [[[-1], [2]],
         [[-3], [4]]],
        dtype=np.int32
    )
    seed = 42
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.array(
        [[[np.nan, 1.0], [np.inf, -np.inf], [0.0, -1.5]],
         [[3.2, -4.1], [5.5, 6.6], [-7.7, 8.8]]],
        dtype=np.float32
    )
    seed = 3
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.arange(2*2*2*3, dtype=np.float64).reshape(2, 2, 2, 3)
    seed = 7
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.array(
        [[[[1, 2, 3, 4],
           [5, 6, 7, 8],
           [9, 10, 11, 12],
           [13, 14, 15, 16]],
          [[-1, -2, -3, -4],
           [-5, -6, -7, -8],
           [-9, -10, -11, -12],
           [-13, -14, -15, -16]],
          [[21, 22, 23, 24],
           [25, 26, 27, 28],
           [29, 30, 31, 32],
           [33, 34, 35, 36]]]],
        dtype=np.int16
    )
    seed = 9
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.array(
        [[[True], [False], [True], [False], [True]],
         [[False], [True], [False], [True], [False]],
         [[True], [True], [False], [False], [True]],
         [[False], [False], [True], [True], [False]]],
        dtype=bool
    )
    seed = 11
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.random.randint(0, 256, size=(3, 5, 2, 1)).astype(np.uint8)
    seed = 13
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.array(
        [[[0.1, -0.2, 0.3],
          [0.4, 0.5, -0.6],
          [0.7, -0.8, 0.9],
          [1.0, -1.1, 1.2],
          [-1.3, 1.4, -1.5]]],
        dtype=np.float16
    )
    seed = 15
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.array(
        [[[[1, 2], [3, 4]]],
         [[[5, 6], [7, 8]]],
         [[[9, 10], [11, 12]]],
         [[[13, 14], [15, 16]]]],
        dtype=np.float32
    ).reshape(4, 1, 2, 2)  # Just to ensure shape is clear; not necessary but keeps intent
    image = np.array(
        [[[1, 2]],
         [[3, 4]],
         [[5, 6]],
         [[7, 8]]],
        dtype=np.float32
    ).reshape(4, 1, 2)
    seed = 17
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.asfortranarray(np.arange(18, dtype=np.float32).reshape(2, 3, 3))
    seed = 19
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.array(
        [[[[1, 2, 3, 4], [5, 6, 7, 8]],
          [[-1, -2, -3, -4], [-5, -6, -7, -8]]],
         [[[9, 10, 11, 12], [13, 14, 15, 16]],
          [[-9, -10, -11, -12], [-13, -14, -15, -16]]]],
        dtype=np.int8
    )
    seed = 21
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.arange(6*7, dtype=np.int64).reshape(6, 7, 1)
    seed = 23
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    return list_of_inputs

generated_inputs["tf.image.random_flip_left_right"] = tf_image_random_flip_left_right_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.random_flip_left_right' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.random_flip_left_right'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.random_flip_left_right', generated_inputs['tf.image.random_flip_left_right'], lib="tf", suffix=0)
