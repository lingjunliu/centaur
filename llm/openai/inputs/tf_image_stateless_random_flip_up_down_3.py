
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_stateless_random_flip_up_down_inputs():
    list_of_inputs = []

    image = np.random.randint(0, 256, size=(4, 4, 3), dtype=np.uint8)
    seed = [123, 456]
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.array([[[1.0], [-2.5]], [[3.3], [4.4]]], dtype=np.float32)
    seed = [0, 1]
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.arange(2 * 3 * 3 * 1).reshape(2, 3, 3, 1).astype(np.float64)
    seed = [42, 24]
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.array([[[[-1, 2], [3, -4], [5, 6]], [[-7, 8], [9, -10], [11, 12]]]], dtype=np.int32)
    seed = [7, 8]
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.arange(3 * 2 * 1).reshape(3, 2, 1).astype(np.uint8)
    seed = [9, 9]
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.random.randint(0, 256, size=(3, 1, 5, 3), dtype=np.uint8)
    seed = [100, 200]
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    arr = np.linspace(-10, 10, num=5 * 4 * 2).astype(np.float32)
    arr[3] = np.nan
    arr[7] = np.inf
    image = arr.reshape(5, 4, 2)
    seed = [555, 666]
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.random.randint(-1000, 1000, size=(2, 4, 4, 4), dtype=np.int64)
    seed = [31415, 92653]
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.array([[[5.5]]], dtype=np.float64)
    seed = [11, 12]
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.random.randn(1, 10, 7, 3).astype(np.float32)
    seed = [333, 444]
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.random.randint(0, 255, size=(6, 5, 1), dtype=np.uint8)
    seed = [2021, 2022]
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    return list_of_inputs

generated_inputs["tf.image.stateless_random_flip_up_down_3"] = tf_image_stateless_random_flip_up_down_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.stateless_random_flip_up_down_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.stateless_random_flip_up_down_3'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.stateless_random_flip_up_down', generated_inputs['tf.image.stateless_random_flip_up_down_3'], lib="tf", suffix=3)
