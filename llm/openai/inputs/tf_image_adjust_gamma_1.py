
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_adjust_gamma_inputs():
    list_of_inputs = []

    image = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    gamma = np.float32(0.5)
    gain = np.float32(1.0)
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
    gamma = np.float32(2.2)
    gain = np.float32(1.0)
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.arange(12, dtype=np.int32).reshape(3, 4, 1)
    gamma = np.float32(1.5)
    gain = np.float32(0.8)
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.array([[-10, 0], [10, 100]], dtype=np.int32)
    gamma = np.float32(1.0)
    gain = np.float32(-1.5)
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.array([[[0.0, 0.2], [0.8, 1.0]]], dtype=np.float64)
    gamma = np.float64(0.2)
    gain = np.float64(0.5)
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.array([0.0, 1.0, 0.5, 0.25], dtype=np.float16)
    gamma = np.float16(5.0)
    gain = np.float16(1.0)
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.arange(2 * 3 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3, 3)
    gamma = np.float32(2.0)
    gain = np.float32(2.0)
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.array([-0.5, 0.0, 0.5, 1.5], dtype=np.float32)
    gamma = np.float32(2.0)
    gain = np.float32(1.0)
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.array([[[[0.1], [0.2]], [[0.3], [0.4]]]], dtype=np.float32)
    gamma = np.float32(0.75)
    gain = np.float32(1.0)
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.array([[[0.1, 0.5, 0.9, 1.0], [0.0, 0.2, 0.4, 0.6]],
                      [[0.3, 0.7, 0.8, 0.2], [0.9, 0.1, 0.5, 0.3]]], dtype=np.float32)
    gamma = np.float32(1.8)
    gain = np.float32(0.7)
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.linspace(1e-6, 1e-2, 5, dtype=np.float32)
    gamma = np.float32(0.1)
    gain = np.float32(1.0)
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.array([[100.0, 500.0, 1000.0], [0.1, 1.0, 10.0]], dtype=np.float32)
    gamma = np.float32(0.3)
    gain = np.float32(1.2)
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = (np.arange(2 * 2 * 3, dtype=np.int32).reshape(2, 2, 3) - 5)
    gamma = np.float32(0.0)
    gain = np.float32(0.5)
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.image.adjust_gamma_1"] = tf_image_adjust_gamma_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.adjust_gamma_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.adjust_gamma_1'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.adjust_gamma', generated_inputs['tf.image.adjust_gamma_1'], lib="tf", suffix=1)
