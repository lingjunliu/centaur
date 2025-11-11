
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_adjust_gamma_inputs():
    list_of_inputs = []

    image1 = np.random.rand(2, 2, 3).astype(np.float32)
    gamma1 = 0.5
    gain1 = 1.2
    input_dict1 = {"image": image1, "gamma": gamma1, "gain": gain1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    image2 = np.random.rand(4, 4, 3).astype(np.float32)
    gamma2 = 2.0
    gain2 = 0.8
    input_dict2 = {"image": image2, "gamma": gamma2, "gain": gain2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    image3 = np.random.rand(1, 1, 3).astype(np.float32)
    gamma3 = 1.0
    gain3 = 1.0
    input_dict3 = {"image": image3, "gamma": gamma3, "gain": gain3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    image4 = np.random.rand(2, 3).astype(np.float32)
    gamma4 = 0.3
    gain4 = 1.5
    input_dict4 = {"image": image4, "gamma": gamma4, "gain": gain4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    image5 = np.random.rand(3, 2, 3).astype(np.float32)
    gamma5 = 1.5
    gain5 = 0.5
    input_dict5 = {"image": image5, "gamma": gamma5, "gain": gain5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    image6 = np.random.rand(5, 5, 3).astype(np.float32)
    gamma6 = 0.0
    gain6 = 2.0
    input_dict6 = {"image": image6, "gamma": gamma6, "gain": gain6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    image7 = np.random.rand(2, 2, 3).astype(np.float32)
    gamma7 = 2.5
    gain7 = 0.2
    input_dict7 = {"image": image7, "gamma": gamma7, "gain": gain7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    image8 = np.random.rand(1, 1, 1).astype(np.float32)
    gamma8 = 0.7
    gain8 = 1.8
    input_dict8 = {"image": image8, "gamma": gamma8, "gain": gain8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    image9 = np.random.rand(4, 3, 3).astype(np.float32)
    gamma9 = 1.1
    gain9 = 0.9
    input_dict9 = {"image": image9, "gamma": gamma9, "gain": gain9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    image10 = np.random.rand(2, 4).astype(np.float32)
    gamma10 = 0.9
    gain10 = 1.1
    input_dict10 = {"image": image10, "gamma": gamma10, "gain": gain10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.image.adjust_gamma"] = tf_image_adjust_gamma_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.adjust_gamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.adjust_gamma'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.adjust_gamma', generated_inputs['tf.image.adjust_gamma'], lib="tf", suffix=0)
