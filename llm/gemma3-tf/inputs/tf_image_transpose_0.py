
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_transpose_inputs():
    list_of_inputs = []

    image_1 = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    input_dict_1 = {"image": tf.constant(image_1), "name": "transpose_1"}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    image_2 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.int32)
    input_dict_2 = {"image": tf.constant(image_2), "name": "transpose_2"}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    image_3 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict_3 = {"image": tf.constant(image_3), "name": "transpose_3"}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    image_4 = np.random.rand(2, 3, 4, 5).astype(np.float32)
    input_dict_4 = {"image": tf.constant(image_4), "name": "transpose_4"}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    image_5 = np.random.randint(-10, 10, size=(1, 5, 5, 3)).astype(np.int32)
    input_dict_5 = {"image": tf.constant(image_5), "name": "transpose_5"}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    image_6 = np.array([[[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]]], dtype=np.float32)
    input_dict_6 = {"image": tf.constant(image_6), "name": "transpose_6"}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    image_7 = np.random.rand(3, 4, 5).astype(np.float32)
    input_dict_7 = {"image": tf.constant(image_7), "name": "transpose_7"}
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    image_8 = np.random.randint(0, 256, size=(2, 2, 3)).astype(np.uint8)
    input_dict_8 = {"image": tf.constant(image_8, dtype=tf.float32), "name": "transpose_8"}
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    image_9 = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    input_dict_9 = {"image": tf.constant(image_9), "name": "transpose_9"}
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    image_10 = np.random.rand(4, 3, 2, 1).astype(np.float32)
    input_dict_10 = {"image": tf.constant(image_10), "name": "transpose_10"}
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.image.transpose"] = tf_image_transpose_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.transpose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.transpose'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.transpose', generated_inputs['tf.image.transpose'], lib="tf", suffix=0)
