
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_stateless_random_flip_up_down_inputs():
    list_of_inputs = []

    image = np.arange(2*3*1, dtype=np.int32).reshape(2, 3, 1)
    seed = np.array([0, 1], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.arange(2*3*2*3, dtype=np.float32).reshape(2, 3, 2, 3) - 10.0
    seed = np.array([2, 3], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = (np.arange(4*4*1, dtype=np.int32) - 50).reshape(4, 4, 1)
    seed = np.array([123, 456], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.array([[[[1.0, np.nan], [np.inf, -np.inf]],
                        [[0.0, -1.0], [np.nan, 5.5]]]], dtype=np.float64)
    seed = np.array([9, 99], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = (np.arange(5*1*2, dtype=np.int32) % 2).reshape(5, 1, 2)
    seed = np.array([42, 24], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.arange(3*5*4*1, dtype=np.int32).reshape(3, 5, 4, 1)
    seed = np.array([7, 11], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.array([[[1234567]]], dtype=np.int32)
    seed = np.array([31415, 92653], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.linspace(-1, 1, 1*1*5*4).astype(np.float32).reshape(1, 1, 5, 4)
    seed = np.array([0, 0], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = (np.arange(7*3*3).reshape(7, 3, 3).astype(np.float32) / 10.0)
    seed = np.array([100, 200], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.arange(2*4*3*2, dtype=np.int32).reshape(2, 4, 3, 2)
    seed = np.array([2147483647, 2147483646], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.arange(10*10*4, dtype=np.float64).reshape(10, 10, 4) / 255.0
    seed = np.array([555, 777], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = ((np.arange(4*2*2*1) % 3) - 1).astype(np.int32).reshape(4, 2, 2, 1)
    seed = np.array([8, 16], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    return list_of_inputs

generated_inputs["tf.image.stateless_random_flip_up_down_1"] = tf_image_stateless_random_flip_up_down_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.stateless_random_flip_up_down_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.stateless_random_flip_up_down_1'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.stateless_random_flip_up_down', generated_inputs['tf.image.stateless_random_flip_up_down_1'], lib="tf", suffix=1)
