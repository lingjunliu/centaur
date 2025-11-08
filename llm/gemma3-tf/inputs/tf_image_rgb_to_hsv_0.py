
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_rgb_to_hsv_inputs():
    list_of_inputs = []

    images1 = np.random.rand(5, 5, 3).astype(np.float32)
    input_dict1 = {"images": tf.convert_to_tensor(images1), "name": "rgb_to_hsv_1"}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    images2 = np.random.rand(2, 3, 3).astype(np.float64)
    input_dict2 = {"images": tf.convert_to_tensor(images2), "name": "rgb_to_hsv_2"}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    images3 = np.random.rand(10, 10, 3).astype(np.float16)
    input_dict3 = {"images": tf.convert_to_tensor(images3), "name": "rgb_to_hsv_3"}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    images4 = np.random.rand(3, 3, 3).astype(np.float32)
    input_dict4 = {"images": tf.convert_to_tensor(images4), "name": "rgb_to_hsv_4"}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    images5 = np.random.rand(4, 4, 3).astype(np.float32)
    input_dict5 = {"images": tf.convert_to_tensor(images5), "name": "rgb_to_hsv_5"}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    images6 = np.random.rand(1, 1, 3).astype(np.float32)
    input_dict6 = {"images": tf.convert_to_tensor(images6), "name": "rgb_to_hsv_6"}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    images7 = np.random.rand(6, 6, 3).astype(np.float32)
    input_dict7 = {"images": tf.convert_to_tensor(images7), "name": "rgb_to_hsv_7"}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    images8 = np.random.rand(7, 7, 3).astype(np.float32)
    input_dict8 = {"images": tf.convert_to_tensor(images8), "name": "rgb_to_hsv_8"}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    images9 = np.random.rand(2, 2, 3).astype(np.float32)
    input_dict9 = {"images": tf.convert_to_tensor(images9), "name": "rgb_to_hsv_9"}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    images10 = np.random.rand(8, 8, 3).astype(np.float32)
    input_dict10 = {"images": tf.convert_to_tensor(images10), "name": "rgb_to_hsv_10"}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.image.rgb_to_hsv"] = tf_image_rgb_to_hsv_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.rgb_to_hsv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.rgb_to_hsv'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.rgb_to_hsv', generated_inputs['tf.image.rgb_to_hsv'], lib="tf", suffix=0)
