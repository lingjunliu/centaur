
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_stateless_random_hue_inputs():
    list_of_inputs = []

    image1 = np.random.rand(2, 2, 3).astype(np.float32)
    max_delta1 = 0.1
    seed1 = np.array([1, 2], dtype=np.int32)
    input_dict1 = {"image": image1, "max_delta": max_delta1, "seed": seed1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    image2 = np.random.rand(4, 4, 3).astype(np.float32)
    max_delta2 = 0.5
    seed2 = np.array([3, 4], dtype=np.int64)
    input_dict2 = {"image": image2, "max_delta": max_delta2, "seed": seed2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    image3 = np.random.rand(1, 1, 3).astype(np.float32)
    max_delta3 = 0.2
    seed3 = np.array([5, 6], dtype=np.int32)
    input_dict3 = {"image": image3, "max_delta": max_delta3, "seed": seed3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    image4 = np.random.rand(3, 3, 3).astype(np.float32)
    max_delta4 = 0.05
    seed4 = np.array([7, 8], dtype=np.int64)
    input_dict4 = {"image": image4, "max_delta": max_delta4, "seed": seed4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    image5 = np.random.rand(2, 2, 3).astype(np.float32)
    max_delta5 = 0.3
    seed5 = np.array([9, 10], dtype=np.int32)
    input_dict5 = {"image": image5, "max_delta": max_delta5, "seed": seed5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    image6 = np.random.rand(5, 5, 3).astype(np.float32)
    max_delta6 = 0.4
    seed6 = np.array([11, 12], dtype=np.int64)
    input_dict6 = {"image": image6, "max_delta": max_delta6, "seed": seed6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    image7 = np.random.rand(1, 5, 3).astype(np.float32)
    max_delta7 = 0.15
    seed7 = np.array([13, 14], dtype=np.int32)
    input_dict7 = {"image": image7, "max_delta": max_delta7, "seed": seed7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    image8 = np.random.rand(5, 1, 3).astype(np.float32)
    max_delta8 = 0.25
    seed8 = np.array([15, 16], dtype=np.int64)
    input_dict8 = {"image": image8, "max_delta": max_delta8, "seed": seed8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    image9 = np.random.rand(3, 4, 3).astype(np.float32)
    max_delta9 = 0.35
    seed9 = np.array([17, 18], dtype=np.int32)
    input_dict9 = {"image": image9, "max_delta": max_delta9, "seed": seed9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    image10 = np.random.rand(4, 3, 3).astype(np.float32)
    max_delta10 = 0.45
    seed10 = np.array([19, 20], dtype=np.int64)
    input_dict10 = {"image": image10, "max_delta": max_delta10, "seed": seed10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

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
