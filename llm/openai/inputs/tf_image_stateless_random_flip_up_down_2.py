
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_stateless_random_flip_up_down_inputs():
    list_of_inputs = []

    image = np.array([[[1], [2]], [[3], [4]]], dtype=np.uint8)
    seed = (np.int32(2), np.int32(3))
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.arange(4 * 3 * 3, dtype=np.float32).reshape(4, 3, 3)
    seed = (0, 0)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.arange(2 * 3 * 4 * 1, dtype=np.int32).reshape(2, 3, 4, 1) - 10
    seed = (123, 456)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.arange(1 * 1 * 5 * 2, dtype=np.float64).reshape(1, 1, 5, 2)
    seed = (999, 1)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = (np.arange(9) % 2 == 0).reshape(3, 3, 1)
    image = base.astype(np.bool_)
    seed = (5, 6)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.random.randn(2, 1, 4, 3).astype(np.float32)
    seed = (7, 8)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = (np.arange(10, dtype=np.int64) - 5).reshape(1, 5, 2)
    seed = (-12, 34)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.empty((0, 3, 3, 3), dtype=np.float32)
    seed = (11, 22)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.arange(10 * 10 * 4, dtype=np.float16).reshape(10, 10, 4)
    seed = (np.int64(2147483647), np.int64(-1))
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = (np.arange(25, dtype=np.float32) / 10.0).reshape(5, 5, 1)
    seed = (42, 24)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.image.stateless_random_flip_up_down_2"] = tf_image_stateless_random_flip_up_down_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.stateless_random_flip_up_down_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.stateless_random_flip_up_down_2'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.stateless_random_flip_up_down', generated_inputs['tf.image.stateless_random_flip_up_down_2'], lib="tf", suffix=2)
