
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_stateless_random_brightness_inputs():
    list_of_inputs = []

    # Input 1: Basic 3D float32 image, standard seed
    image_1 = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                        [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    max_delta_1 = 0.2
    seed_1 = np.array([1, 2], dtype=np.int32)
    list_of_inputs.append({
        'image': image_1,
        'max_delta': max_delta_1,
        'seed': seed_1
    })

    # Input 2: 4D float32 batch of images, int32 seed
    image_2 = np.random.rand(2, 3, 3, 3).astype(np.float32)
    max_delta_2 = 0.5
    seed_2 = np.array([42, 24], dtype=np.int32)
    list_of_inputs.append({
        'image': image_2,
        'max_delta': max_delta_2,
        'seed': seed_2
    })

    # Input 3: 3D uint8 image, int64 seed (Note: on XLA, only int32 is allowed, so using int32 is safer, but standard supports int64 too. Let's use int32 for general safety, or mix them.)
    image_3 = np.random.randint(0, 256, size=(10, 10, 3), dtype=np.uint8)
    max_delta_3 = 10.0
    seed_3 = np.array([123, 456], dtype=np.int32)
    list_of_inputs.append({
        'image': image_3,
        'max_delta': max_delta_3,
        'seed': seed_3
    })

    # Input 4: 3D float16 grayscale image, small max_delta
    image_4 = np.random.rand(4, 4, 1).astype(np.float16)
    max_delta_4 = 0.1
    seed_4 = np.array([9, 99], dtype=np.int32)
    list_of_inputs.append({
        'image': image_4,
        'max_delta': max_delta_4,
        'seed': seed_4
    })

    # Input 5: 4D uint8 batch of grayscale images, large max_delta
    image_5 = np.random.randint(0, 256, size=(3, 5, 5, 1), dtype=np.uint8)
    max_delta_5 = 50.0
    seed_5 = np.array([7, 7], dtype=np.int32)
    list_of_inputs.append({
        'image': image_5,
        'max_delta': max_delta_5,
        'seed': seed_5
    })

    # Input 6: Zero max_delta (no brightness adjustment), 3D float32 image
    image_6 = np.random.rand(100, 100, 3).astype(np.float32)
    max_delta_6 = 0.0
    seed_6 = np.array([0, 0], dtype=np.int32)
    list_of_inputs.append({
        'image': image_6,
        'max_delta': max_delta_6,
        'seed': seed_6
    })

    # Input 7: 4D float32 RGBA images (4 channels)
    image_7 = np.random.rand(2, 10, 10, 4).astype(np.float32)
    max_delta_7 = 0.3
    seed_7 = np.array([11, 22], dtype=np.int32)
    list_of_inputs.append({
        'image': image_7,
        'max_delta': max_delta_7,
        'seed': seed_7
    })

    # Input 8: Minimal sized 3D float32 image (1x1x1)
    image_8 = np.array([[[0.5]]], dtype=np.float32)
    max_delta_8 = 1.5
    seed_8 = np.array([1000, 2000], dtype=np.int32)
    list_of_inputs.append({
        'image': image_8,
        'max_delta': max_delta_8,
        'seed': seed_8
    })

    # Input 9: 4D uint8 batch, standard seed
    image_9 = np.random.randint(0, 256, size=(4, 8, 8, 3), dtype=np.uint8)
    max_delta_9 = 25.0
    seed_9 = np.array([111, 222], dtype=np.int32)
    list_of_inputs.append({
        'image': image_9,
        'max_delta': max_delta_9,
        'seed': seed_9
    })

    # Input 10: 3D float32 image, small max_delta
    image_10 = np.random.rand(5, 5, 3).astype(np.float32)
    max_delta_10 = 0.05
    seed_10 = np.array([12345, 67890], dtype=np.int32)
    list_of_inputs.append({
        'image': image_10,
        'max_delta': max_delta_10,
        'seed': seed_10
    })

    return list_of_inputs

generated_inputs["tf.image.stateless_random_brightness"] = tf_image_stateless_random_brightness_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.stateless_random_brightness' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.stateless_random_brightness'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.stateless_random_brightness', generated_inputs['tf.image.stateless_random_brightness'], lib="tf", suffix=0)
