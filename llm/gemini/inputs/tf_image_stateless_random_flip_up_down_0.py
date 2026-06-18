
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_stateless_random_flip_up_down_inputs():
    list_of_inputs = []

    # Input 1: 3-D float32 image, int32 seed
    image_1 = np.random.rand(4, 4, 3).astype(np.float32)
    seed_1 = np.array([1, 2], dtype=np.int32)
    list_of_inputs.append({"image": image_1, "seed": seed_1})

    # Input 2: 3-D float32 image, int64 seed
    image_2 = np.random.rand(8, 8, 3).astype(np.float32)
    seed_2 = np.array([42, 43], dtype=np.int64)
    list_of_inputs.append({"image": image_2, "seed": seed_2})

    # Input 3: 4-D float32 image (batch size 2), int32 seed
    image_3 = np.random.rand(2, 6, 6, 1).astype(np.float32)
    seed_3 = np.array([123, 456], dtype=np.int32)
    list_of_inputs.append({"image": image_3, "seed": seed_3})

    # Input 4: 4-D int32 image, int32 seed
    image_4 = np.random.randint(-100, 100, size=(1, 5, 5, 3)).astype(np.int32)
    seed_4 = np.array([7, 8], dtype=np.int32)
    list_of_inputs.append({"image": image_4, "seed": seed_4})

    # Input 5: 3-D float64 image, int64 seed
    image_5 = np.random.rand(10, 10, 4).astype(np.float64)
    seed_5 = np.array([99, 100], dtype=np.int64)
    list_of_inputs.append({"image": image_5, "seed": seed_5})

    # Input 6: 4-D int32 image, int32 seed
    image_6 = np.random.randint(-1000, 1000, size=(3, 3, 3, 2)).astype(np.int32)
    seed_6 = np.array([0, 0], dtype=np.int32)
    list_of_inputs.append({"image": image_6, "seed": seed_6})

    # Input 7: 3-D float32 image, int64 seed
    image_7 = np.random.rand(1, 1, 1).astype(np.float32)
    seed_7 = np.array([-1, -2], dtype=np.int64)
    list_of_inputs.append({"image": image_7, "seed": seed_7})

    # Input 8: 4-D float32 image, int32 seed
    image_8 = np.random.rand(4, 16, 16, 3).astype(np.float32)
    seed_8 = np.array([9999, 8888], dtype=np.int32)
    list_of_inputs.append({"image": image_8, "seed": seed_8})

    # Input 9: 3-D float64 image, int32 seed
    image_9 = np.random.rand(7, 7, 2).astype(np.float64)
    seed_9 = np.array([111, 222], dtype=np.int32)
    list_of_inputs.append({"image": image_9, "seed": seed_9})

    # Input 10: 4-D int64 image, int64 seed
    image_10 = np.random.randint(0, 100000, size=(2, 8, 8, 4)).astype(np.int64)
    seed_10 = np.array([123456789, 987654321], dtype=np.int64)
    list_of_inputs.append({"image": image_10, "seed": seed_10})

    return list_of_inputs

generated_inputs["tf.image.stateless_random_flip_up_down"] = tf_image_stateless_random_flip_up_down_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.stateless_random_flip_up_down' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.stateless_random_flip_up_down'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.stateless_random_flip_up_down', generated_inputs['tf.image.stateless_random_flip_up_down'], lib="tf", suffix=0)
