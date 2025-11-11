
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_MatrixSetDiag_inputs():
    list_of_inputs = []

    input1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    diagonal1 = np.array([[1, 0], [0, 1]], dtype=np.float32)
    input_dict1 = {'name': 'diag1', 'input': input1, 'diagonal': diagonal1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.int32)
    diagonal2 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    input_dict2 = {'name': 'diag2', 'input': input2, 'diagonal': diagonal2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array([[[1, 2], [3, 4]]], dtype=np.float32)
    diagonal3 = np.array([[1.0]], dtype=np.float32)
    input_dict3 = {'name': 'diag3', 'input': input3, 'diagonal': diagonal3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    return list_of_inputs

generated_inputs["tf.raw_ops.MatrixSetDiag"] = tf_raw_ops_MatrixSetDiag_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.MatrixSetDiag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MatrixSetDiag'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.MatrixSetDiag', generated_inputs['tf.raw_ops.MatrixSetDiag'], lib="tf", suffix=0)
