
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_stateless_random_crop_inputs():
    list_of_inputs = []

    value = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
    size = np.array([4, 4, 3], dtype=np.int32)
    seed = np.array([123, 456], dtype=np.int32)
    name = "crop_rgb_uint8"
    list_of_inputs.append(copy.deepcopy({"value": value, "size": size, "seed": seed, "name": name}))

    value = np.random.randn(10, 15).astype(np.float32)
    size = np.array([5, 7], dtype=np.int32)
    seed = np.array([1, 2], dtype=np.int32)
    name = "crop_2d_float32"
    list_of_inputs.append(copy.deepcopy({"value": value, "size": size, "seed": seed, "name": name}))

    value = np.arange(20, dtype=np.int32)
    size = np.array([10], dtype=np.int32)
    seed = np.array([42, 24], dtype=np.int32)
    name = "crop_1d_int32"
    list_of_inputs.append(copy.deepcopy({"value": value, "size": size, "seed": seed, "name": name}))

    value = np.random.rand(5, 32, 32, 3).astype(np.float64)
    size = np.array([3, 16, 16, 3], dtype=np.int32)
    seed = np.array([7, 11], dtype=np.int32)
    name = "crop_4d_batch"
    list_of_inputs.append(copy.deepcopy({"value": value, "size": size, "seed": seed, "name": name}))

    value = np.random.randint(-100, 100, (6, 4, 1), dtype=np.int32)
    size = np.array([3, 2, 1], dtype=np.int32)
    seed = np.array([101, 202], dtype=np.int32)
    name = "crop_neg_ints"
    list_of_inputs.append(copy.deepcopy({"value": value, "size": size, "seed": seed, "name": name}))

    value = np.random.randn(2, 3, 4, 5, 6).astype(np.float64)
    size = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    seed = np.array([31415, 92653], dtype=np.int32)
    name = "crop_5d_float64"
    list_of_inputs.append(copy.deepcopy({"value": value, "size": size, "seed": seed, "name": name}))

    value = (np.random.rand(9, 9, 1) > 0.5).astype(np.bool_)
    size = np.array([5, 5, 1], dtype=np.int32)
    seed = np.array([0, 1], dtype=np.int32)
    name = "crop_bool_mask"
    list_of_inputs.append(copy.deepcopy({"value": value, "size": size, "seed": seed, "name": name}))

    value = np.random.rand(4, 3, 28, 28).astype(np.float32)
    size = np.array([2, 3, 14, 14], dtype=np.int32)
    seed = np.array([1234, 5678], dtype=np.int32)
    name = "crop_channels_first"
    list_of_inputs.append(copy.deepcopy({"value": value, "size": size, "seed": seed, "name": name}))

    value = np.random.randint(0, 10, (7, 11), dtype=np.int16)
    size = np.array([7, 11], dtype=np.int32)
    seed = np.array([9, 99], dtype=np.int32)
    name = "crop_full_nochange"
    list_of_inputs.append(copy.deepcopy({"value": value, "size": size, "seed": seed, "name": name}))

    value = np.random.randint(0, 5, (12, 12, 4), dtype=np.int8)
    size = np.array([12, 8, 4], dtype=np.int32)
    seed = np.array([555, 666], dtype=np.int32)
    name = "crop_reduce_width"
    list_of_inputs.append(copy.deepcopy({"value": value, "size": size, "seed": seed, "name": name}))

    value = np.random.rand(3, 6, 7, 2).astype(np.float16)
    size = np.array([3, 5, 6, 2], dtype=np.int32)
    seed = np.array([2021, 2022], dtype=np.int32)
    name = "crop_float16"
    list_of_inputs.append(copy.deepcopy({"value": value, "size": size, "seed": seed, "name": name}))

    value = np.random.randint(-1000, 1000, (2, 5), dtype=np.int64)
    size = np.array([1, 4], dtype=np.int32)
    seed = np.array([888888, 999999], dtype=np.int32)
    name = "crop_2d_int64"
    list_of_inputs.append(copy.deepcopy({"value": value, "size": size, "seed": seed, "name": name}))

    return list_of_inputs

generated_inputs["tf.image.stateless_random_crop"] = tf_image_stateless_random_crop_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.stateless_random_crop' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.stateless_random_crop'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.stateless_random_crop', generated_inputs['tf.image.stateless_random_crop'], lib="tf", suffix=0)
