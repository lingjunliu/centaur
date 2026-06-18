
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_stateless_random_flip_left_right_inputs():
    list_of_inputs = []
    
    # Input 1: 3-D float32 image, standard seed
    list_of_inputs.append({
        'image': np.random.rand(2, 2, 1).astype(np.float32),
        'seed': np.array([1, 2], dtype=np.int32)
    })
    
    # Input 2: 3-D uint8 image, different seed
    list_of_inputs.append({
        'image': np.random.randint(0, 256, (3, 4, 3), dtype=np.uint8),
        'seed': np.array([42, 24], dtype=np.int32)
    })
    
    # Input 3: 4-D float32 image, standard seed
    list_of_inputs.append({
        'image': np.random.rand(1, 2, 2, 1).astype(np.float32),
        'seed': np.array([10, 20], dtype=np.int32)
    })
    
    # Input 4: 3-D int32 image with negative values
    list_of_inputs.append({
        'image': np.random.randint(-100, 100, (3, 3, 3), dtype=np.int32),
        'seed': np.array([0, 0], dtype=np.int32)
    })
    
    # Input 5: 4-D int32 image, negative values
    list_of_inputs.append({
        'image': np.random.randint(-50, 50, (2, 5, 5, 3), dtype=np.int32),
        'seed': np.array([5, 10], dtype=np.int32)
    })
    
    # Input 6: 3-D float32 image with 1 channel
    list_of_inputs.append({
        'image': np.random.rand(4, 4, 1).astype(np.float32),
        'seed': np.array([99, 99], dtype=np.int32)
    })
    
    # Input 7: 3-D float32 minimal image size
    list_of_inputs.append({
        'image': np.random.rand(1, 1, 1).astype(np.float32),
        'seed': np.array([123, 456], dtype=np.int32)
    })
    
    # Input 8: 4-D int32 image with single pixel height
    list_of_inputs.append({
        'image': np.random.randint(0, 10, (2, 1, 3, 2), dtype=np.int32),
        'seed': np.array([7, 8], dtype=np.int32)
    })
    
    # Input 9: 3-D float32 negative values only
    list_of_inputs.append({
        'image': (np.random.rand(3, 2, 2) * -10).astype(np.float32),
        'seed': np.array([100, 200], dtype=np.int32)
    })
    
    # Input 10: 4-D uint8 image
    list_of_inputs.append({
        'image': np.random.randint(0, 256, (2, 2, 2, 2), dtype=np.uint8),
        'seed': np.array([1000, 2000], dtype=np.int32)
    })
    
    return list_of_inputs

generated_inputs["tf.image.stateless_random_flip_left_right"] = tf_image_stateless_random_flip_left_right_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.stateless_random_flip_left_right' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.stateless_random_flip_left_right'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.stateless_random_flip_left_right', generated_inputs['tf.image.stateless_random_flip_left_right'], lib="tf", suffix=0)
