
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_MatrixSetDiagV2_inputs():
    list_of_inputs = []

    # Input 1: Basic main diagonal replacement
    input_np = np.array([[7, 7, 7], [7, 7, 7], [7, 7, 7]]).astype(np.int32)
    diagonal_np = np.array([1, 2, 3]).astype(np.int32)
    k_np = np.array(0).astype(np.int32)

    input_dict = {"input": input_np, "diagonal": diagonal_np, "k": k_np, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Superdiagonal replacement (k=1)
    input_np = np.array([[7, 7, 7], [7, 7, 7], [7, 7, 7]]).astype(np.int32)
    diagonal_np = np.array([1, 2]).astype(np.int32)
    k_np = np.array(1).astype(np.int32)

    input_dict = {"input": input_np, "diagonal": diagonal_np, "k": k_np, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Subdiagonal replacement (k=-1)
    input_np = np.array([[7, 7, 7], [7, 7, 7], [7, 7, 7]]).astype(np.int32)
    diagonal_np = np.array([1, 2]).astype(np.int32)
    k_np = np.array(-1).astype(np.int32)

    input_dict = {"input": input_np, "diagonal": diagonal_np, "k": k_np, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.MatrixSetDiagV2"] = tf_raw_ops_MatrixSetDiagV2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MatrixSetDiagV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MatrixSetDiagV2'.")

check_valid('tf.raw_ops.MatrixSetDiagV2', generated_inputs['tf.raw_ops.MatrixSetDiagV2'], lib="tf", suffix=0)
