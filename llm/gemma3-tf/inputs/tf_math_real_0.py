
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_real_inputs():
    list_of_inputs = []

    input1 = np.array([-2.25 + 4.75j, 3.25 + 5.75j])
    input_dict1 = {"input": tf.constant(input1), "name": "real_part1"}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([1 + 0j, 2 + 0j, 3 + 0j])
    input_dict2 = {"input": tf.constant(input2), "name": "real_part2"}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]])
    input_dict3 = {"input": tf.constant(input3), "name": "real_part3"}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[-1.5 - 2.5j], [3.5 - 4.5j]])
    input_dict4 = {"input": tf.constant(input4), "name": "real_part4"}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([5.0 + 0.0j])
    input_dict5 = {"input": tf.constant(input5), "name": "real_part5"}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[-1 + 2j, 3 - 4j], [5 + 6j, -7 - 8j]])
    input_dict6 = {"input": tf.constant(input6), "name": "real_part6"}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([1j, 2j, 3j])
    input_dict7 = {"input": tf.constant(input7), "name": "real_part7"}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([[[1 + 1j], [2 + 2j]], [[3 + 3j], [4 + 4j]]])
    input_dict8 = {"input": tf.constant(input8), "name": "real_part8"}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([-1.0 - 1.0j, -2.0 - 2.0j])
    input_dict9 = {"input": tf.constant(input9), "name": "real_part9"}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([0 + 1j, 0 - 1j])
    input_dict10 = {"input": tf.constant(input10), "name": "real_part10"}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.math.real"] = tf_math_real_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.real' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.real'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.real', generated_inputs['tf.math.real'], lib="tf", suffix=0)
