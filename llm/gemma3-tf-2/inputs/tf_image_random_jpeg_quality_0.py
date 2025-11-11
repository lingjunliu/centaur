
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_random_jpeg_quality_inputs():
    list_of_inputs = []

    image1 = np.random.randint(0, 256, size=(2, 2, 3), dtype=np.uint8)
    min_jpeg_quality1 = 75
    max_jpeg_quality1 = 95
    seed1 = 42
    input_dict1 = {
        "image": image1,
        "min_jpeg_quality": min_jpeg_quality1,
        "max_jpeg_quality": max_jpeg_quality1,
        "seed": seed1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    image2 = np.random.randint(0, 256, size=(4, 4, 3), dtype=np.uint8)
    min_jpeg_quality2 = 20
    max_jpeg_quality2 = 80
    seed2 = 100
    input_dict2 = {
        "image": image2,
        "min_jpeg_quality": min_jpeg_quality2,
        "max_jpeg_quality": max_jpeg_quality2,
        "seed": seed2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    image3 = np.random.randint(0, 256, size=(1, 5, 3), dtype=np.uint8)
    min_jpeg_quality3 = 1
    max_jpeg_quality3 = 100
    seed3 = 0
    input_dict3 = {
        "image": image3,
        "min_jpeg_quality": min_jpeg_quality3,
        "max_jpeg_quality": max_jpeg_quality3,
        "seed": seed3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    image4 = np.random.randint(0, 256, size=(3, 3, 3), dtype=np.uint8)
    min_jpeg_quality4 = 50
    max_jpeg_quality4 = 70
    seed4 = 123
    input_dict4 = {
        "image": image4,
        "min_jpeg_quality": min_jpeg_quality4,
        "max_jpeg_quality": max_jpeg_quality4,
        "seed": seed4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    image5 = np.random.randint(0, 256, size=(2, 4, 3), dtype=np.uint8)
    min_jpeg_quality5 = 90
    max_jpeg_quality5 = 99
    seed5 = 50
    input_dict5 = {
        "image": image5,
        "min_jpeg_quality": min_jpeg_quality5,
        "max_jpeg_quality": max_jpeg_quality5,
        "seed": seed5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
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
