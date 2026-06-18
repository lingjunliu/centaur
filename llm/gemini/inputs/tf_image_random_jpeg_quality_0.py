
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_random_jpeg_quality_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'image': np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8),
        'min_jpeg_quality': 50,
        'max_jpeg_quality': 90,
        'seed': 42
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'image': np.random.randint(0, 256, size=(50, 50, 1), dtype=np.uint8),
        'min_jpeg_quality': 10,
        'max_jpeg_quality': 20,
        'seed': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'image': (np.random.rand(200, 150, 3) * 255.0).astype(np.float32),
        'min_jpeg_quality': 75,
        'max_jpeg_quality': 95,
        'seed': 1234
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'image': np.random.rand(10, 10, 1).astype(np.float32),
        'min_jpeg_quality': 0,
        'max_jpeg_quality': 100,
        'seed': 10
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'image': np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8),
        'min_jpeg_quality': 30,
        'max_jpeg_quality': 60,
        'seed': 99
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'image': np.random.randint(0, 256, size=(32, 32, 1), dtype=np.uint8),
        'min_jpeg_quality': 80,
        'max_jpeg_quality': 90,
        'seed': 111
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'image': (np.random.rand(128, 128, 3) * 255.0).astype(np.float32),
        'min_jpeg_quality': 45,
        'max_jpeg_quality': 55,
        'seed': 7
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'image': np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8),
        'min_jpeg_quality': 20,
        'max_jpeg_quality': 80,
        'seed': 456
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'image': np.random.randint(0, 256, size=(3, 3, 1), dtype=np.uint8),
        'min_jpeg_quality': 1,
        'max_jpeg_quality': 2,
        'seed': 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'image': (np.random.rand(512, 512, 3) * 255.0).astype(np.float32),
        'min_jpeg_quality': 90,
        'max_jpeg_quality': 100,
        'seed': 888
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
