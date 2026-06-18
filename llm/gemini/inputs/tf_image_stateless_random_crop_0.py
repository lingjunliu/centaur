
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_stateless_random_crop_inputs():
    list_of_inputs = []

    # Input 1: 3D float32 image
    input_dict = {
        "value": np.random.randn(10, 10, 3).astype(np.float32),
        "size": np.array([5, 5, 3], dtype=np.int32),
        "seed": np.array([1, 2], dtype=np.int32),
        "name": "crop_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D uint8 image (e.g. MNIST size)
    input_dict = {
        "value": np.random.randint(0, 256, size=(28, 28, 1)).astype(np.uint8),
        "size": np.array([14, 14, 1], dtype=np.int32),
        "seed": np.array([42, 43], dtype=np.int32),
        "name": "crop_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 4D batch of float64 images
    input_dict = {
        "value": np.random.randn(4, 32, 32, 3).astype(np.float64),
        "size": np.array([2, 16, 16, 3], dtype=np.int32),
        "seed": np.array([10, 20], dtype=np.int32),
        "name": "crop_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D integer array
    input_dict = {
        "value": np.random.randint(-100, 100, size=(100, 100)).astype(np.int32),
        "size": np.array([50, 50], dtype=np.int32),
        "seed": np.array([100, 200], dtype=np.int32),
        "name": "crop_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D float32 vector
    input_dict = {
        "value": np.random.randn(10).astype(np.float32),
        "size": np.array([5], dtype=np.int32),
        "seed": np.array([0, 0], dtype=np.int32),
        "name": "crop_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 5D video-like tensor
    input_dict = {
        "value": np.random.randn(2, 5, 16, 16, 3).astype(np.float32),
        "size": np.array([1, 3, 8, 8, 3], dtype=np.int32),
        "seed": np.array([9, 9], dtype=np.int32),
        "name": "crop_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Crop same size as input (identity crop)
    input_dict = {
        "value": np.random.randint(0, 100, size=(8, 8, 3)).astype(np.int64),
        "size": np.array([8, 8, 3], dtype=np.int32),
        "seed": np.array([7, 8], dtype=np.int32),
        "name": "crop_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Crop with size 1 along each dimension
    input_dict = {
        "value": np.random.randn(5, 5, 5).astype(np.float32),
        "size": np.array([1, 1, 1], dtype=np.int32),
        "seed": np.array([123, 456], dtype=np.int32),
        "name": "crop_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large values, using int32 size and seed
    input_dict = {
        "value": np.random.randn(12, 12, 4).astype(np.float64),
        "size": np.array([6, 6, 2], dtype=np.int32),
        "seed": np.array([55, 66], dtype=np.int32),
        "name": "crop_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Int16 high resolution image
    input_dict = {
        "value": np.random.randint(-32768, 32767, size=(256, 256, 3)).astype(np.int16),
        "size": np.array([128, 128, 3], dtype=np.int32),
        "seed": np.array([1111, 2222], dtype=np.int32),
        "name": "crop_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

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
