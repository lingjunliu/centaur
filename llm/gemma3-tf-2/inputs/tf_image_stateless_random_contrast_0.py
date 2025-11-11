
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_stateless_random_contrast_inputs():
    list_of_inputs = []

    image1 = np.random.rand(2, 2, 3).astype(np.float32)
    seed1 = np.array([1, 2], dtype=np.int32)
    input_dict1 = {
        "image": image1,
        "lower": 0.2,
        "upper": 0.5,
        "seed": seed1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    image2 = np.random.rand(4, 4, 3).astype(np.float32)
    seed2 = np.array([3, 4], dtype=np.int32)
    input_dict2 = {
        "image": image2,
        "lower": 0.0,
        "upper": 0.1,
        "seed": seed2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    image3 = np.random.rand(1, 5, 5, 3).astype(np.float32)
    seed3 = np.array([5, 6], dtype=np.int32)
    input_dict3 = {
        "image": image3,
        "lower": 0.0,
        "upper": 1.0,
        "seed": seed3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    image4 = np.random.rand(2, 2, 2, 3).astype(np.float32)
    seed4 = np.array([7, 8], dtype=np.int32)
    input_dict4 = {
        "image": image4,
        "lower": 0.5,
        "upper": 1.0,
        "seed": seed4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    image5 = np.random.rand(5, 5, 3).astype(np.float32)
    seed5 = np.array([9, 10], dtype=np.int32)
    input_dict5 = {
        "image": image5,
        "lower": 0.0,
        "upper": 0.2,
        "seed": seed5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

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
