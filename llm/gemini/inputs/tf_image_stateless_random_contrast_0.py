
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_stateless_random_contrast_inputs():
    list_of_inputs = []

    # Input 1: Basic Float32, 3D
    list_of_inputs.append({
        'image': np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                           [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32),
        'lower': 0.2,
        'upper': 0.5,
        'seed': np.array([1, 2], dtype=np.int32)
    })

    # Input 2: Large dimensions, Float32, 3D
    list_of_inputs.append({
        'image': np.random.rand(10, 10, 3).astype(np.float32),
        'lower': 0.1,
        'upper': 0.9,
        'seed': np.array([42, 43], dtype=np.int64)
    })

    # Input 3: Float32, 1-channel, 3D
    list_of_inputs.append({
        'image': np.random.rand(5, 5, 1).astype(np.float32),
        'lower': 0.0,
        'upper': 1.0,
        'seed': np.array([0, 0], dtype=np.int32)
    })

    # Input 4: Float32, 3D with 3 channels
    list_of_inputs.append({
        'image': np.random.rand(4, 4, 3).astype(np.float32),
        'lower': 0.5,
        'upper': 1.5,
        'seed': np.array([123, 456], dtype=np.int32)
    })

    # Input 5: Batched Float32, 4D
    list_of_inputs.append({
        'image': np.random.rand(2, 8, 8, 3).astype(np.float32),
        'lower': 0.2,
        'upper': 1.8,
        'seed': np.array([7, 8], dtype=np.int64)
    })

    # Input 6: Float32, 3D, small size
    list_of_inputs.append({
        'image': np.random.rand(3, 3, 3).astype(np.float32),
        'lower': 0.5,
        'upper': 0.6,
        'seed': np.array([10, 20], dtype=np.int32)
    })

    # Input 7: High scale factors, Float32, 3D
    list_of_inputs.append({
        'image': np.random.rand(6, 6, 3).astype(np.float32),
        'lower': 2.0,
        'upper': 5.0,
        'seed': np.array([99, 999], dtype=np.int32)
    })

    # Input 8: 5D Tensor, Float32
    list_of_inputs.append({
        'image': np.random.rand(2, 2, 4, 4, 3).astype(np.float32),
        'lower': 0.8,
        'upper': 1.2,
        'seed': np.array([1, 1], dtype=np.int64)
    })

    # Input 9: Batched Float32, 4D, single channel
    list_of_inputs.append({
        'image': np.random.rand(1, 16, 16, 1).astype(np.float32),
        'lower': 0.0,
        'upper': 2.0,
        'seed': np.array([4, 2], dtype=np.int32)
    })

    # Input 10: Float32 image, 3D, very close bounds
    list_of_inputs.append({
        'image': np.random.rand(3, 3, 3).astype(np.float32),
        'lower': 1.0,
        'upper': 1.0001,
        'seed': np.array([11, 22], dtype=np.int32)
    })

    return list_of_inputs

generated_inputs["tf.image.stateless_random_contrast"] = tf_image_stateless_random_contrast_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.stateless_random_contrast' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.stateless_random_contrast'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.stateless_random_contrast', generated_inputs['tf.image.stateless_random_contrast'], lib="tf", suffix=0)
