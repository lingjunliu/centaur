
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_diagpart_inputs():
    list_of_inputs = []

    input1 = np.array([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]], dtype=np.float32)
    input_dict1 = {'name': 'diag_part_1', 'input': input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[[1, 0], [0, 2]], [[0, 3], [4, 0]]], dtype=np.int32)
    input_dict2 = {'name': 'diag_part_2', 'input': input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[[1, -1], [-1, 1]], [[-1, 1], [1, -1]]], dtype=np.float64)
    input_dict3 = {'name': 'diag_part_3', 'input': input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[[1, 0, 0], [0, 2, 0], [0, 0, 3]], [[0, 0, 0], [0, 4, 0], [0, 0, 5]]], dtype=np.complex128)
    input_dict4 = {'name': 'diag_part_4', 'input': input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]], [[0, 0, 0, 0], [0, 5, 0, 0], [0, 0, 6, 0], [0, 0, 0, 7]]], dtype=np.float32)
    input_dict5 = {'name': 'diag_part_5', 'input': input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[[1, 0], [0, 2]], [[0, 3], [4, 0]]], dtype=np.half)
    input_dict6 = {'name': 'diag_part_6', 'input': input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([[[1, 0, 0], [0, 2, 0], [0, 0, 3]]], dtype=np.int64)
    input_dict7 = {'name': 'diag_part_7', 'input': input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([[[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]]], dtype=np.float32)
    input_dict8 = {'name': 'diag_part_8', 'input': input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = np.array([[[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]],[[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]]], dtype=np.float32)
    input_dict9 = {'name': 'diag_part_9', 'input': input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = np.array([[[1,0],[0,2],[0,0]],[[0,0],[3,0],[0,4]]], dtype=np.int32)
    input_dict10 = {'name': 'diag_part_10', 'input': input10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.raw_ops.DiagPart"] = tf_raw_ops_diagpart_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.DiagPart' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DiagPart'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.DiagPart', generated_inputs['tf.raw_ops.DiagPart'], lib="tf", suffix=0)
