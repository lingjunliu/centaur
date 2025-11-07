
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_stateless_random_flip_left_right_inputs():
    list_of_inputs = []

    image1 = np.array([[[1], [2]], [[3], [4]]], dtype=np.int32)
    seed1 = np.array([2, 3], dtype=np.int32)
    input_dict1 = {"image": image1, "seed": seed1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    image2 = np.random.randint(0, 256, size=(2, 3, 4, 1), dtype=np.int32)
    seed2 = np.array([5, 7], dtype=np.int32)
    input_dict2 = {"image": image2, "seed": seed2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    image3 = np.random.randint(-100, 100, size=(1, 5, 6, 3), dtype=np.int32)
    seed3 = np.array([10, 20], dtype=np.int32)
    input_dict3 = {"image": image3, "seed": seed3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    image4 = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.int64)
    seed4 = np.array([1, 1], dtype=np.int64)
    input_dict4 = {"image": image4, "seed": seed4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    image5 = np.random.randint(0, 256, size=(4, 4, 3), dtype=np.int32)
    seed5 = np.array([123, 456], dtype=np.int32)
    input_dict5 = {"image": image5, "seed": seed5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    image6 = np.random.randint(0, 256, size=(1, 1, 1, 1), dtype=np.int32)
    seed6 = np.array([789, 101], dtype=np.int32)
    input_dict6 = {"image": image6, "seed": seed6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    image7 = np.random.randint(-50, 50, size=(2, 2, 2, 2), dtype=np.int32)
    seed7 = np.array([2, 2], dtype=np.int32)
    input_dict7 = {"image": image7, "seed": seed7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    image8 = np.random.randint(0, 256, size=(3, 3, 3), dtype=np.int32)
    seed8 = np.array([99, 100], dtype=np.int32)
    input_dict8 = {"image": image8, "seed": seed8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    image9 = np.random.randint(0, 256, size=(5, 5, 1, 4), dtype=np.int32)
    seed9 = np.array([11, 12], dtype=np.int32)
    input_dict9 = {"image": image9, "seed": seed9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    image10 = np.random.randint(-200, 200, size=(1, 6, 7, 2), dtype=np.int32)
    seed10 = np.array([13, 14], dtype=np.int32)
    input_dict10 = {"image": image10, "seed": seed10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    

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
