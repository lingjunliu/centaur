
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_stateless_random_hue_inputs():
    list_of_inputs = []

    # Case 1: 3D float32 image, max_delta 0.2, int32 seed
    image = np.random.rand(2, 2, 3).astype(np.float32)
    max_delta = 0.2
    seed = np.array([1, 2], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'max_delta': max_delta,
        'seed': seed
    })

    # Case 2: 3D float32 image, max_delta 0.0 (no change), int64 seed
    image = np.random.rand(4, 4, 3).astype(np.float32)
    max_delta = 0.0
    seed = np.array([42, 42], dtype=np.int64)
    list_of_inputs.append({
        'image': image,
        'max_delta': max_delta,
        'seed': seed
    })

    # Case 3: 3D float32 image (1x1 pixel), max_delta 0.5, int32 seed
    image = np.random.rand(1, 1, 3).astype(np.float32)
    max_delta = 0.5
    seed = np.array([0, 0], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'max_delta': max_delta,
        'seed': seed
    })

    # Case 4: 4D float32 image batch, max_delta 0.1, int32 seed
    image = np.random.rand(2, 3, 4, 3).astype(np.float32)
    max_delta = 0.1
    seed = np.array([99, 100], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'max_delta': max_delta,
        'seed': seed
    })

    # Case 5: 3D float32 image, max_delta 0.3, int64 seed
    image = np.random.rand(1, 3, 3).astype(np.float32)
    max_delta = 0.3
    seed = np.array([123, 456], dtype=np.int64)
    list_of_inputs.append({
        'image': image,
        'max_delta': max_delta,
        'seed': seed
    })

    # Case 6: Larger 3D float32 image, max_delta 0.45, int32 seed
    image = np.random.rand(32, 32, 3).astype(np.float32)
    max_delta = 0.45
    seed = np.array([7, 8], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'max_delta': max_delta,
        'seed': seed
    })

    # Case 7: 4D float32 image, max_delta 0.05, int64 seed
    image = np.random.rand(2, 2, 2, 3).astype(np.float32)
    max_delta = 0.05
    seed = np.array([11, 22], dtype=np.int64)
    list_of_inputs.append({
        'image': image,
        'max_delta': max_delta,
        'seed': seed
    })

    # Case 8: 3D float32 image, max_delta 0.15, int32 seed
    image = np.random.rand(5, 5, 3).astype(np.float32)
    max_delta = 0.15
    seed = np.array([2023, 2024], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'max_delta': max_delta,
        'seed': seed
    })

    # Case 9: 3D float32 image, max_delta 0.5, negative int32 seed
    image = np.random.rand(2, 3, 3).astype(np.float32)
    max_delta = 0.5
    seed = np.array([-1, -2], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'max_delta': max_delta,
        'seed': seed
    })

    # Case 10: 3D float32 image, max_delta 0.25, large int64 seed
    image = np.random.rand(1, 2, 3).astype(np.float32)
    max_delta = 0.25
    seed = np.array([123456, 789012], dtype=np.int64)
    list_of_inputs.append({
        'image': image,
        'max_delta': max_delta,
        'seed': seed
    })

    return list_of_inputs

generated_inputs["tf.image.stateless_random_hue"] = tf_image_stateless_random_hue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.stateless_random_hue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.stateless_random_hue'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.stateless_random_hue', generated_inputs['tf.image.stateless_random_hue'], lib="tf", suffix=0)
