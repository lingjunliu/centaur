
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_add_inputs():
    list_of_inputs = []

    # Input 1: SparseTensor + Dense Tensor, threshold = 0
    a_indices = np.array([[0, 0], [1, 2]])
    a_values = np.array([1, 2], dtype=np.float32)
    a_shape = np.array([2, 3])
    a = tf.SparseTensor(indices=a_indices, values=a_values, dense_shape=a_shape)
    b = tf.constant([[3, 0, 1], [0, 2, 0]], dtype=np.float32)
    threshold = tf.constant(0.0, dtype=np.float32)

    input_dict = {"a": a, "b": b, "threshold": threshold}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.add"] = tf_sparse_add_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sparse.add' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.add'.")

check_valid('tf.sparse.add', generated_inputs['tf.sparse.add'], lib="tf", suffix=0)
