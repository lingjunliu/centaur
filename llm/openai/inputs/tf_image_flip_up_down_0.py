
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_flip_up_down_inputs():
    list_of_inputs = []

    image = np.array(
        [[[1.0, 2.0, 3.0],
          [4.0, 5.0, 6.0]],
         [[7.0, 8.0, 9.0],
          [10.0, 11.0, 12.0]]],
        dtype=np.float32
    )
    list_of_inputs.append(copy.deepcopy({"image": image}))

    image = np.arange(2 * 3 * 4 * 1, dtype=np.float32).reshape(2, 3, 4, 1)
    list_of_inputs.append(copy.deepcopy({"image": image}))

    image = np.array(
        [[[-1], [0], [1]],
         [[2], [-3], [4]],
         [[5], [6], [-7]]],
        dtype=np.int32
    )
    list_of_inputs.append(copy.deepcopy({"image": image}))

    image = np.array(
        [[[[255, 0, 0, 255],
           [0, 255, 0, 255],
           [0, 0, 255, 255]],
          [[10, 20, 30, 40],
           [50, 60, 70, 80],
           [90, 100, 110, 120]]]],
        dtype=np.uint8
    )
    list_of_inputs.append(copy.deepcopy({"image": image}))

    image = np.array(
        [[[True, False],
          [False, True]],
         [[True, True],
          [False, False]]],
        dtype=bool
    )
    list_of_inputs.append(copy.deepcopy({"image": image}))

    image = np.linspace(-1.5, 1.5, num=2 * 2 * 3 * 2, dtype=np.float16).reshape(2, 2, 3, 2)
    list_of_inputs.append(copy.deepcopy({"image": image}))

    image = np.array(
        [[[0.1], [0.2]],
         [[0.3], [0.4]],
         [[0.5], [0.6]],
         [[0.7], [0.8]]],
        dtype=np.float64
    )
    list_of_inputs.append(copy.deepcopy({"image": image}))

    image = np.arange(-36, -36 + 3 * 2 * 2 * 3, dtype=np.int16).reshape(3, 2, 2, 3)
    list_of_inputs.append(copy.deepcopy({"image": image}))

    image = np.array([[[1, -2, 3, -4, 5],
                       [6, -7, 8, -9, 10]]], dtype=np.int8)
    list_of_inputs.append(copy.deepcopy({"image": image}))

    image = (np.arange(1 * 5 * 2 * 2, dtype=np.float32).reshape(1, 5, 2, 2) / 10.0)
    list_of_inputs.append(copy.deepcopy({"image": image}))

    image = np.arange(2 * 3 * 1, dtype=np.int64).reshape(2, 3, 1)
    list_of_inputs.append(copy.deepcopy({"image": image}))

    return list_of_inputs

generated_inputs["tf.image.flip_up_down"] = tf_image_flip_up_down_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.flip_up_down' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.flip_up_down'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.flip_up_down', generated_inputs['tf.image.flip_up_down'], lib="tf", suffix=0)
