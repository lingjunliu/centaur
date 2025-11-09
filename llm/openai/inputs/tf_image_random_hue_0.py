
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_random_hue_inputs():
    list_of_inputs = []

    # Input 1
    image = np.random.rand(4, 4, 3).astype(np.float32)
    max_delta = np.float32(0.2)
    seed = np.int32(123)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 2
    image = np.random.rand(2, 4, 4, 3).astype(np.float32)
    max_delta = np.float64(0.5)
    seed = np.int64(0)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 3
    image = np.random.rand(1, 1, 3).astype(np.float64)
    max_delta = np.float64(0.0)
    seed = np.int32(7)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 4
    image = np.random.rand(8, 5, 3).astype(np.float32)
    max_delta = np.float32(0.15)
    seed = np.int32(9999)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 5
    image = np.linspace(0.0, 1.0, num=3*2*3, dtype=np.float32).reshape(3, 2, 3)
    max_delta = np.float32(0.05)
    seed = np.int32(2021)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 6
    image = np.random.rand(2, 3, 4, 3).astype(np.float32)
    max_delta = np.float32(0.49)
    seed = np.int32(314159)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 7
    image = np.vstack([
        np.zeros((3, 3, 3), dtype=np.float32),
        np.ones((3, 3, 3), dtype=np.float32)
    ])
    max_delta = np.float32(0.25)
    seed = np.int32(1)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 8
    image = np.random.rand(2, 2, 2, 2, 3).astype(np.float64)
    max_delta = np.float64(0.3)
    seed = np.int64(123456789)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 9
    image = np.random.rand(1, 7, 3).astype(np.float32)
    max_delta = np.float32(0.000001)
    seed = np.int32(4242)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 10
    image = np.random.randint(0, 256, size=(10, 10, 3), dtype=np.uint8)
    max_delta = np.float32(0.4)
    seed = np.int32(0)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 11
    image = np.random.randint(0, 256, size=(3, 5, 5, 3), dtype=np.uint8)
    max_delta = np.float32(0.12)
    seed = np.int32(555)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 12
    image = np.tile(np.eye(3, dtype=np.float32)[None, :, :], (4, 1, 1))
    max_delta = np.float32(0.35)
    seed = np.int32(8888)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    return list_of_inputs

generated_inputs["tf.image.random_hue"] = tf_image_random_hue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.random_hue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.random_hue'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.random_hue', generated_inputs['tf.image.random_hue'], lib="tf", suffix=0)
