
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_random_hue_inputs():
    list_of_inputs = []

    # Input 1
    image = np.random.rand(2, 2, 3).astype(np.float32)
    max_delta = 0.2
    seed = 42
    list_of_inputs.append({
        'image': image,
        'max_delta': max_delta,
        'seed': seed
    })

    # Input 2
    image = np.random.randint(0, 256, size=(10, 10, 3)).astype(np.uint8)
    max_delta = 0.1
    seed = 123
    list_of_inputs.append({
        'image': image,
        'max_delta': max_delta,
        'seed': seed
    })

    # Input 3
    image = np.random.rand(2, 5, 5, 3).astype(np.float32)
    max_delta = 0.5
    seed = 1
    list_of_inputs.append({
        'image': image,
        'max_delta': max_delta,
        'seed': seed
    })

    # Input 4
    image = np.random.rand(1, 1, 3).astype(np.float32)
    max_delta = 0.0
    seed = 0
    list_of_inputs.append({
        'image': image,
        'max_delta': max_delta,
        'seed': seed
    })

    # Input 5
    image = np.random.randint(0, 256, size=(1, 3, 3, 3)).astype(np.uint8)
    max_delta = 0.3
    seed = 999
    list_of_inputs.append({
        'image': image,
        'max_delta': max_delta,
        'seed': seed
    })

    # Input 6
    image = np.random.rand(100, 100, 3).astype(np.float32)
    max_delta = 0.45
    seed = 88
    list_of_inputs.append({
        'image': image,
        'max_delta': max_delta,
        'seed': seed
    })

    # Input 7
    image = np.random.rand(5, 8, 8, 3).astype(np.float32)
    max_delta = 0.05
    seed = 7
    list_of_inputs.append({
        'image': image,
        'max_delta': max_delta,
        'seed': seed
    })

    # Input 8
    image = np.random.randint(0, 256, size=(32, 32, 3)).astype(np.uint8)
    max_delta = 0.25
    seed = 456
    list_of_inputs.append({
        'image': image,
        'max_delta': max_delta,
        'seed': seed
    })

    # Input 9
    image = np.random.randint(0, 256, size=(3, 16, 16, 3)).astype(np.uint8)
    max_delta = 0.15
    seed = 777
    list_of_inputs.append({
        'image': image,
        'max_delta': max_delta,
        'seed': seed
    })

    # Input 10
    image = np.random.rand(4, 4, 3).astype(np.float32)
    max_delta = 0.35
    seed = 11
    list_of_inputs.append({
        'image': image,
        'max_delta': max_delta,
        'seed': seed
    })

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
