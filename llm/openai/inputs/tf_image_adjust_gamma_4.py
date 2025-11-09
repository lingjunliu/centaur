
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import torch
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)
np.random.seed(42)

def tf_image_adjust_gamma_inputs():
    list_of_inputs = []

    image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    gamma = np.array(2.2, dtype=np.float32)
    gain = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = np.random.rand(32, 32, 3).astype(np.float32)
    gamma = np.array(0.5, dtype=np.float32)
    gain = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = np.random.rand(48, 48).astype(np.float32)
    gamma = np.array(1.5, dtype=np.float32)
    gain = np.array(0.8, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = np.random.rand(2, 32, 32, 1).astype(np.float32)
    gamma = np.array(0.8, dtype=np.float32)
    gain = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = np.random.rand(4, 16, 16, 3).astype(np.float32)
    gamma = np.array(1.2, dtype=np.float32)
    gain = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
    gamma = np.array(1.0, dtype=np.float32)
    gain = np.array(2.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = np.random.rand(1, 10, 10, 4).astype(np.float32)
    gamma = np.array(0.9, dtype=np.float32)
    gain = np.array(1.1, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = np.random.rand(256, 256, 3).astype(np.float16)
    gamma = np.array(0.6, dtype=np.float16)
    gain = np.array(1.0, dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = np.random.randint(0, 256, (5, 5, 3), dtype=np.uint8)
    gamma = np.array(0.0, dtype=np.float32)
    gain = np.array(128.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = np.linspace(0.0, 1.0, num=9, dtype=np.float32).reshape(3, 3)
    gamma = np.array(0.9, dtype=np.float32)
    gain = np.array(0.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = np.random.rand(3, 4, 4, 3).astype(np.float32)
    gamma = np.array(1.1, dtype=np.float32)
    gain = np.array(0.8, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = np.linspace(-1.0, 1.0, num=48, dtype=np.float32).reshape(4, 4, 3)
    gamma = np.array(2.0, dtype=np.float32)
    gain = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    return list_of_inputs

generated_inputs["tf.image.adjust_gamma_4"] = tf_image_adjust_gamma_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.adjust_gamma_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.adjust_gamma_4'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.adjust_gamma', generated_inputs['tf.image.adjust_gamma_4'], lib="tf", suffix=4)
