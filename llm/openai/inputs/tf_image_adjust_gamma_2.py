
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)
np.random.seed(42)

def tf_image_adjust_gamma_inputs():
    list_of_inputs = []

    image = (np.random.rand(8, 8, 3) * 255).astype(np.uint8)
    gamma = np.array(2.2, dtype=np.float32)
    gain = 1.0
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = (np.random.rand(2, 2, 3) * 2.0).astype(np.float32)
    gamma = np.array(0.5, dtype=np.float32)
    gain = 1.0
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = (np.random.rand(2, 3, 4, 3)).astype(np.float32)
    gamma = np.array(0.6, dtype=np.float32)
    gain = 1.0
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = np.array([[[-3000, 0, 3000],
                       [10000, -20000, 15000]],
                      [[-12345, 23456, -1000],
                       [12345, -500, 0]]], dtype=np.int16)
    gamma = np.array(2.0, dtype=np.float32)
    gain = 1.0
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = (np.arange(2 * 3 * 1).reshape(2, 3, 1) + 1).astype(np.int32)
    gamma = np.array(1.0, dtype=np.float32)
    gain = 1.0
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = (np.random.rand(1, 5, 5, 1)).astype(np.float16)
    gamma = np.array(1.3, dtype=np.float16)
    gain = 0.5
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = (np.random.rand(4, 5, 1) * 255).astype(np.uint8)
    gamma = np.array(0.8, dtype=np.float32)
    gain = 0.7
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = (np.random.rand(3, 4, 4, 2)).astype(np.float32)
    gamma = np.array(1.5, dtype=np.float32)
    gain = 0.9
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = (np.random.rand(5, 6, 4)).astype(np.float32)
    gamma = np.array(0.2, dtype=np.float32)
    gain = 1.0
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = np.zeros((6, 6, 2), dtype=np.float32)
    image[0, 0, :] = 0.5
    gamma = np.array(1e-3, dtype=np.float32)
    gain = 2.0
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = np.full((2, 3, 3), 0.5, dtype=np.float64)
    gamma = np.array(2.2, dtype=np.float64)
    gain = 1.0
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = (np.random.rand(7, 5) * 255).astype(np.uint8)
    gamma = np.array(1.8, dtype=np.float32)
    gain = 1.2
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    return list_of_inputs

generated_inputs["tf.image.adjust_gamma_2"] = tf_image_adjust_gamma_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.adjust_gamma_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.adjust_gamma_2'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.adjust_gamma', generated_inputs['tf.image.adjust_gamma_2'], lib="tf", suffix=2)
