
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

np.random.seed(42)

def tf_image_adjust_gamma_inputs():
    list_of_inputs = []

    image = np.random.randint(0, 256, size=(3, 3, 3), dtype=np.uint8)
    gamma = np.float32(0.5)
    gain = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = (np.random.rand(2, 2, 1) * 2.0).astype(np.float32)
    gamma = np.float32(2.2)
    gain = np.array([0.8], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = np.random.randint(0, 1000, size=(5,), dtype=np.int32)
    gamma = np.float32(1.0)
    gain = np.linspace(0.5, 1.5, 5).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = np.random.rand(2, 3, 4, 3).astype(np.float32)
    gamma = np.float32(0.0)
    gain = np.array(0.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = np.random.randint(0, 101, size=(4, 4), dtype=np.int32)
    gamma = np.float32(0.8)
    gain = np.array([[0.5], [1.0], [1.5], [2.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = np.random.rand(5, 5, 1).astype(np.float32)
    gamma = np.float32(1.5)
    gain = np.array(-1.2, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = np.arange(1 * 2 * 3 * 4, dtype=np.uint8).reshape(1, 2, 3, 4)
    gamma = np.float32(3.0)
    gain = np.array([[[[0.5, 1.0, 1.5, 2.0]]]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = np.random.rand(6, 6, 3).astype(np.float32)
    gamma = np.float32(0.2)
    gain = (np.random.rand(6, 6, 3) * 2.0).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = np.arange(15, dtype=np.uint8).reshape(3, 5)
    gamma = np.float32(0.9)
    gain = np.array(2.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = np.array([0.0, 1.0], dtype=np.float32)
    gamma = np.float32(0.7)
    gain = np.array([1.0, 2.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    return list_of_inputs

generated_inputs["tf.image.adjust_gamma_3"] = tf_image_adjust_gamma_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.adjust_gamma_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.adjust_gamma_3'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.adjust_gamma', generated_inputs['tf.image.adjust_gamma_3'], lib="tf", suffix=3)
