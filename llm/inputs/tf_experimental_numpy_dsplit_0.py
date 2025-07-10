
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_dsplit_inputs():
    list_of_inputs = []

    # Input 1
    ary = np.arange(16.0).reshape(2, 2, 4)
    indices_or_sections = 2
    input_dict = {"ary": tf.convert_to_tensor(np.array(ary, dtype=np.float32)), "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    ary = np.arange(27.0).reshape(3, 3, 3)
    indices_or_sections = 3
    input_dict = {"ary": tf.convert_to_tensor(np.array(ary, dtype=np.float32)), "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    ary = np.arange(64.0).reshape(4, 4, 4)
    indices_or_sections = 4
    input_dict = {"ary": tf.convert_to_tensor(np.array(ary, dtype=np.float32)), "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    ary = np.arange(24.0).reshape(2, 3, 4)
    indices_or_sections = 2
    input_dict = {"ary": tf.convert_to_tensor(np.array(ary, dtype=np.float32)), "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    ary = np.arange(24.0).reshape(4, 2, 3)
    indices_or_sections = 3
    input_dict = {"ary": tf.convert_to_tensor(np.array(ary, dtype=np.float32)), "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    ary = np.arange(48.0).reshape(2, 4, 6)
    indices_or_sections = 2
    input_dict = {"ary": tf.convert_to_tensor(np.array(ary, dtype=np.float32)), "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    ary = np.arange(48.0).reshape(2, 4, 6)
    indices_or_sections = 3
    input_dict = {"ary": tf.convert_to_tensor(np.array(ary, dtype=np.float32)), "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    ary = np.arange(48.0).reshape(2, 4, 6)
    indices_or_sections = 6
    input_dict = {"ary": tf.convert_to_tensor(np.array(ary, dtype=np.float32)), "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    ary = np.arange(72.0).reshape(3, 4, 6)
    indices_or_sections = 2
    input_dict = {"ary": tf.convert_to_tensor(np.array(ary, dtype=np.float32)), "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    ary = np.arange(120.0).reshape(5, 4, 6)
    indices_or_sections = 3
    input_dict = {"ary": tf.convert_to_tensor(np.array(ary, dtype=np.float32)), "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.dsplit"] = tf_experimental_numpy_dsplit_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.dsplit' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.dsplit'.")

check_valid('tf.experimental.numpy.dsplit', generated_inputs['tf.experimental.numpy.dsplit'], lib="tf", suffix=0)
