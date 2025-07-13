
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_MatrixSetDiag_inputs():
    list_of_inputs = []

    # Input 1: Basic 2x2 matrix
    input_matrix = np.array([[1, 2], [3, 4]], dtype=np.int32)
    diagonal = np.array([5, 6], dtype=np.int32)
    input_dict = {"input": input_matrix, "diagonal": diagonal, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic 3x3 matrix
    input_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    diagonal = np.array([10, 11, 12], dtype=np.float32)
    input_dict = {"input": input_matrix, "diagonal": diagonal, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.MatrixSetDiag"] = tf_raw_ops_MatrixSetDiag_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MatrixSetDiag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MatrixSetDiag'.")

check_valid('tf.raw_ops.MatrixSetDiag', generated_inputs['tf.raw_ops.MatrixSetDiag'], lib="tf", suffix=0)
