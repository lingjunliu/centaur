
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_stateless_random_flip_up_down_inputs():
    list_of_inputs = []

    image1 = np.array([[[1], [2]], [[3], [4]]])
    seed1 = np.array([2, 3], dtype=np.int32)
    input_dict1 = {"image": image1, "seed": seed1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    image2 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    seed2 = np.array([5, 7], dtype=np.int64)
    input_dict2 = {"image": image2, "seed": seed2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    image3 = np.random.rand(1, 100, 100, 3).astype(np.float32)
    seed3 = np.array([10, 20], dtype=np.int32)
    input_dict3 = {"image": image3, "seed": seed3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    image4 = np.random.randint(0, 256, size=(2, 50, 50, 1), dtype=np.uint8)
    seed4 = np.array([15, 25], dtype=np.int32)
    input_dict4 = {"image": image4, "seed": seed4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    image5 = np.array([[[1, 2, 3]]])
    seed5 = np.array([8, 9], dtype=np.int32)
    input_dict5 = {"image": image5, "seed": seed5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    image6 = np.random.rand(4, 64, 64, 3).astype(np.float32)
    seed6 = np.array([1, 1], dtype=np.int32)
    input_dict6 = {"image": image6, "seed": seed6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    image7 = np.random.randint(0, 100, size=(1, 32, 32, 3), dtype=np.int32)
    seed7 = np.array([100, 200], dtype=np.int32)
    input_dict7 = {"image": image7, "seed": seed7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    image8 = np.array([[[[1, 2], [3, 4]]]])
    seed8 = np.array([123, 456], dtype=np.int32)
    input_dict8 = {"image": image8, "seed": seed8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    image9 = np.random.rand(2, 128, 128, 3).astype(np.float32)
    seed9 = np.array([7, 11], dtype=np.int32)
    input_dict9 = {"image": image9, "seed": seed9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    image10 = np.random.randint(0, 256, size=(3, 64, 64, 1), dtype=np.uint8)
    seed10 = np.array([111, 222], dtype=np.int32)
    input_dict10 = {"image": image10, "seed": seed10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

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
