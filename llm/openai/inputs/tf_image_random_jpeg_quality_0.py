
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_image_random_jpeg_quality_inputs():
    list_of_inputs = []
    rng = np.random.default_rng(12345)

    image = np.array([[[1, 2, 3],
                       [4, 5, 6]],
                      [[7, 8, 9],
                       [10, 11, 12]]], dtype=np.uint8)
    min_jpeg_quality = np.int32(75)
    max_jpeg_quality = np.int32(95)
    seed = np.int32(42)
    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = (np.arange(16, dtype=np.uint8).reshape(4, 4, 1) * 16) % 256
    min_jpeg_quality = np.int32(0)
    max_jpeg_quality = np.int32(100)
    seed = np.int32(0)
    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.array([[[128, 64, 32]]], dtype=np.uint8)
    min_jpeg_quality = np.int32(1)
    max_jpeg_quality = np.int32(2)
    seed = np.int32(7)
    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = rng.integers(0, 256, size=(8, 16, 3), dtype=np.uint8)
    min_jpeg_quality = np.int64(50)
    max_jpeg_quality = np.int64(51)
    seed = np.int64(123)
    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = rng.integers(0, 256, size=(7, 7, 1), dtype=np.uint8)
    min_jpeg_quality = np.int64(10)
    max_jpeg_quality = np.int64(90)
    seed = np.int64(999)
    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    min_jpeg_quality = np.int32(30)
    max_jpeg_quality = np.int32(31)
    seed = np.int32(12)
    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.ones((32, 48, 3), dtype=np.uint8) * 255
    min_jpeg_quality = np.int64(5)
    max_jpeg_quality = np.int64(6)
    seed = np.int64(987654321)
    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = np.fromfunction(lambda i, j: (i * 17 + j * 29) % 256, (15, 15), dtype=int).astype(np.uint8)
    image = base[..., None]
    min_jpeg_quality = np.int32(60)
    max_jpeg_quality = np.int32(80)
    seed = np.int32(31415)
    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
    min_jpeg_quality = np.int32(95)
    max_jpeg_quality = np.int32(100)
    seed = np.int32(271828)
    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = (np.arange(15, dtype=np.uint8).reshape(3, 5, 1) * 17) % 256
    min_jpeg_quality = 20
    max_jpeg_quality = 21
    seed = 2021
    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = ((np.indices((10, 10)).sum(axis=0) % 2) * 255).astype(np.uint8)
    ch1 = base
    ch2 = (base // 2).astype(np.uint8)
    ch3 = (255 - base).astype(np.uint8)
    image = np.stack([ch1, ch2, ch3], axis=-1)
    min_jpeg_quality = 2
    max_jpeg_quality = 10
    seed = 444
    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    big = rng.integers(0, 256, size=(20, 20, 3), dtype=np.uint8)
    image = big[2:15, 3:20, :]
    min_jpeg_quality = 40
    max_jpeg_quality = 70
    seed = 8888
    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.image.random_jpeg_quality"] = tf_image_random_jpeg_quality_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.random_jpeg_quality' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.random_jpeg_quality'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.random_jpeg_quality', generated_inputs['tf.image.random_jpeg_quality'], lib="tf", suffix=0)
