
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_stateless_random_saturation_inputs():
    list_of_inputs = []
    
    # Input 1
    image = np.random.rand(2, 2, 3).astype(np.float32)
    lower = 0.5
    upper = 1.0
    seed = np.array([1, 2], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'lower': lower,
        'upper': upper,
        'seed': seed
    })

    # Input 2
    image = np.random.rand(3, 3, 3).astype(np.float32)
    lower = 0.1
    upper = 0.9
    seed = np.array([42, 43], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'lower': lower,
        'upper': upper,
        'seed': seed
    })

    # Input 3
    image = np.random.randint(0, 256, size=(4, 4, 3)).astype(np.uint8)
    lower = 0.0
    upper = 2.0
    seed = np.array([10, 20], dtype=np.int64)
    list_of_inputs.append({
        'image': image,
        'lower': lower,
        'upper': upper,
        'seed': seed
    })

    # Input 4
    image = np.random.rand(2, 2, 2, 3).astype(np.float64)
    lower = 0.2
    upper = 0.8
    seed = np.array([9, 9], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'lower': lower,
        'upper': upper,
        'seed': seed
    })

    # Input 5
    image = np.random.rand(5, 5, 3).astype(np.float32)
    lower = 1.0
    upper = 3.0
    seed = np.array([0, 1], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'lower': lower,
        'upper': upper,
        'seed': seed
    })

    # Input 6
    image = np.random.rand(1, 1, 3).astype(np.float32)
    lower = 0.0
    upper = 1.0
    seed = np.array([123, 456], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'lower': lower,
        'upper': upper,
        'seed': seed
    })

    # Input 7
    image = np.random.rand(2, 3, 4, 3).astype(np.float32)
    lower = 0.5
    upper = 0.6
    seed = np.array([100, 200], dtype=np.int64)
    list_of_inputs.append({
        'image': image,
        'lower': lower,
        'upper': upper,
        'seed': seed
    })

    # Input 8
    image = np.random.randint(0, 256, size=(5, 5, 3)).astype(np.uint8)
    lower = 1.2
    upper = 1.8
    seed = np.array([7, 8], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'lower': lower,
        'upper': upper,
        'seed': seed
    })

    # Input 9
    image = np.random.rand(2, 1, 1, 3).astype(np.float32)
    lower = 0.1
    upper = 10.0
    seed = np.array([1234, 5678], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'lower': lower,
        'upper': upper,
        'seed': seed
    })

    # Input 10
    image = np.random.rand(3, 3, 3).astype(np.float64)
    lower = 0.0
    upper = 0.5
    seed = np.array([999, 999], dtype=np.int64)
    list_of_inputs.append({
        'image': image,
        'lower': lower,
        'upper': upper,
        'seed': seed
    })

    return list_of_inputs

generated_inputs["tf.image.stateless_random_saturation"] = tf_image_stateless_random_saturation_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.stateless_random_saturation' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.stateless_random_saturation'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.stateless_random_saturation', generated_inputs['tf.image.stateless_random_saturation'], lib="tf", suffix=0)
