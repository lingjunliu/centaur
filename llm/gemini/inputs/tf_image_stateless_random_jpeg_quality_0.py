
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_stateless_random_jpeg_quality_inputs():
    list_of_inputs = []

    # Input 1
    image = np.random.randint(0, 256, size=(10, 10, 3), dtype=np.uint8)
    min_jpeg_quality = 75
    max_jpeg_quality = 95
    seed = np.array([1, 2], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'min_jpeg_quality': min_jpeg_quality,
        'max_jpeg_quality': max_jpeg_quality,
        'seed': seed
    })

    # Input 2
    image = np.random.randint(0, 256, size=(100, 100, 1), dtype=np.uint8)
    min_jpeg_quality = 0
    max_jpeg_quality = 10
    seed = np.array([42, 43], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'min_jpeg_quality': min_jpeg_quality,
        'max_jpeg_quality': max_jpeg_quality,
        'seed': seed
    })

    # Input 3
    image = np.random.rand(32, 32, 3).astype(np.float32)
    min_jpeg_quality = 50
    max_jpeg_quality = 60
    seed = np.array([10, 20], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'min_jpeg_quality': min_jpeg_quality,
        'max_jpeg_quality': max_jpeg_quality,
        'seed': seed
    })

    # Input 4
    image = np.random.randint(0, 256, size=(1, 1, 3), dtype=np.uint8)
    min_jpeg_quality = 90
    max_jpeg_quality = 100
    seed = np.array([100, 200], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'min_jpeg_quality': min_jpeg_quality,
        'max_jpeg_quality': max_jpeg_quality,
        'seed': seed
    })

    # Input 5
    image = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
    min_jpeg_quality = 30
    max_jpeg_quality = 80
    seed = np.array([-5, 5], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'min_jpeg_quality': min_jpeg_quality,
        'max_jpeg_quality': max_jpeg_quality,
        'seed': seed
    })

    # Input 6
    image = np.random.rand(64, 64, 1).astype(np.float32)
    min_jpeg_quality = 10
    max_jpeg_quality = 90
    seed = np.array([1000, 2000], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'min_jpeg_quality': min_jpeg_quality,
        'max_jpeg_quality': max_jpeg_quality,
        'seed': seed
    })

    # Input 7
    image = np.random.randint(0, 256, size=(5, 5, 3), dtype=np.uint8)
    min_jpeg_quality = 0
    max_jpeg_quality = 100
    seed = np.array([0, 0], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'min_jpeg_quality': min_jpeg_quality,
        'max_jpeg_quality': max_jpeg_quality,
        'seed': seed
    })

    # Input 8
    image = np.random.randint(0, 256, size=(128, 128, 3), dtype=np.uint8)
    min_jpeg_quality = 45
    max_jpeg_quality = 55
    seed = np.array([11, 22], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'min_jpeg_quality': min_jpeg_quality,
        'max_jpeg_quality': max_jpeg_quality,
        'seed': seed
    })

    # Input 9
    image = np.random.randint(0, 256, size=(50, 50, 1), dtype=np.uint8)
    min_jpeg_quality = 20
    max_jpeg_quality = 40
    seed = np.array([99, 99], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'min_jpeg_quality': min_jpeg_quality,
        'max_jpeg_quality': max_jpeg_quality,
        'seed': seed
    })

    # Input 10
    image = np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)
    min_jpeg_quality = 80
    max_jpeg_quality = 85
    seed = np.array([1234, 5678], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'min_jpeg_quality': min_jpeg_quality,
        'max_jpeg_quality': max_jpeg_quality,
        'seed': seed
    })

    return list_of_inputs

generated_inputs["tf.image.stateless_random_jpeg_quality"] = tf_image_stateless_random_jpeg_quality_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.stateless_random_jpeg_quality' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.stateless_random_jpeg_quality'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.stateless_random_jpeg_quality', generated_inputs['tf.image.stateless_random_jpeg_quality'], lib="tf", suffix=0)
