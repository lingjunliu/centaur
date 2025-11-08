
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_UnicodeScript_inputs():
    list_of_inputs = []

    input1 = np.array([1, 31, 38], dtype=np.int32)
    input_dict1 = {'name': 'test1', 'input': input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([65, 97, 122], dtype=np.int32)
    input_dict2 = {'name': 'test2', 'input': input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([0, 65535, 10000], dtype=np.int32)
    input_dict3 = {'name': 'test3', 'input': input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([-1, 256, 512], dtype=np.int32)
    input_dict4 = {'name': 'test4', 'input': input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict5 = {'name': 'test5', 'input': input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    input_dict6 = {'name': 'test6', 'input': input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)
    input_dict7 = {'name': 'test7', 'input': input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([1000, 2000, 3000], dtype=np.int32)
    input_dict8 = {'name': 'test8', 'input': input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([66, 67, 68, 69], dtype=np.int32)
    input_dict9 = {'name': 'test9', 'input': input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    input_dict10 = {'name': 'test10', 'input': input10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.raw_ops.UnicodeScript"] = tf_raw_ops_UnicodeScript_inputs()


def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.UnicodeScript' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.UnicodeScript'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.UnicodeScript', generated_inputs['tf.raw_ops.UnicodeScript'], lib="tf", suffix=0)
