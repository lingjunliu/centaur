
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_sparse_add_inputs():
    list_of_inputs = []

    # Input 1: SparseTensor + Tensor
    indices1 = np.array([[0, 0], [1, 2]], dtype=np.int64)
    values1 = np.array([1, 2], dtype=np.float32)
    shape1 = np.array([2, 3], dtype=np.int64)
    a = tf.SparseTensor(indices1, values1, shape1)
    b = np.array([[3, 0, 1], [0, 4, 0]], dtype=np.float32)
    threshold = np.float32(0.0)
    input_dict = {"a": a, "b": b, "threshold": threshold}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Tensor + SparseTensor
    a = np.array([[3, 0, 1], [0, 4, 0]], dtype=np.float32)
    indices2 = np.array([[0, 0], [1, 2]], dtype=np.int64)
    values2 = np.array([1, 2], dtype=np.float32)
    shape2 = np.array([2, 3], dtype=np.int64)
    b = tf.SparseTensor(indices2, values2, shape2)
    threshold = np.float32(0.0)
    input_dict = {"a": a, "b": b, "threshold": threshold}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: SparseTensor + SparseTensor, threshold = 0
    indices3_a = np.array([[0, 0], [1, 2]], dtype=np.int64)
    values3_a = np.array([1, 2], dtype=np.float32)
    shape3_a = np.array([2, 3], dtype=np.int64)
    a = tf.SparseTensor(indices3_a, values3_a, shape3_a)
    indices3_b = np.array([[0, 0], [1, 1]], dtype=np.int64)
    values3_b = np.array([-1, 3], dtype=np.float32)
    shape3_b = np.array([2, 3], dtype=np.int64)
    b = tf.SparseTensor(indices3_b, values3_b, shape3_b)
    threshold = np.float32(0.0)
    input_dict = {"a": a, "b": b, "threshold": threshold}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: SparseTensor + SparseTensor, threshold > 0
    indices4_a = np.array([[0, 0], [1, 2]], dtype=np.int64)
    values4_a = np.array([1, 2], dtype=np.float32)
    shape4_a = np.array([2, 3], dtype=np.int64)
    a = tf.SparseTensor(indices4_a, values4_a, shape4_a)
    indices4_b = np.array([[0, 0], [1, 1]], dtype=np.int64)
    values4_b = np.array([-1, 3], dtype=np.float32)
    shape4_b = np.array([2, 3], dtype=np.int64)
    b = tf.SparseTensor(indices4_b, values4_b, shape4_b)
    threshold = np.float32(0.5)
    input_dict = {"a": a, "b": b, "threshold": threshold}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: SparseTensor + SparseTensor, complex values
    indices5_a = np.array([[0, 0], [1, 2]], dtype=np.int64)
    values5_a = np.array([1 + 1j, 2 - 2j], dtype=np.complex64)
    shape5_a = np.array([2, 3], dtype=np.int64)
    a = tf.SparseTensor(indices5_a, values5_a, shape5_a)
    indices5_b = np.array([[0, 0], [1, 1]], dtype=np.int64)
    values5_b = np.array([-1 - 1j, 3 + 3j], dtype=np.complex64)
    shape5_b = np.array([2, 3], dtype=np.int64)
    b = tf.SparseTensor(indices5_b, values5_b, shape5_b)
    threshold = np.float32(0.5)
    input_dict = {"a": a, "b": b, "threshold": threshold}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: SparseTensor + Tensor, complex values
    indices6 = np.array([[0, 0], [1, 2]], dtype=np.int64)
    values6 = np.array([1 + 1j, 2 - 2j], dtype=np.complex64)
    shape6 = np.array([2, 3], dtype=np.int64)
    a = tf.SparseTensor(indices6, values6, shape6)
    b = np.array([[3 - 1j, 0, 1], [0, 4 + 2j, 0]], dtype=np.complex64)
    threshold = np.float32(0.0)
    input_dict = {"a": a, "b": b, "threshold": threshold}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensor + SparseTensor, complex values
    a = np.array([[3 - 1j, 0, 1], [0, 4 + 2j, 0]], dtype=np.complex64)
    indices7 = np.array([[0, 0], [1, 2]], dtype=np.int64)
    values7 = np.array([1 + 1j, 2 - 2j], dtype=np.complex64)
    shape7 = np.array([2, 3], dtype=np.int64)
    b = tf.SparseTensor(indices7, values7, shape7)
    threshold = np.float32(0.0)
    input_dict = {"a": a, "b": b, "threshold": threshold}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D SparseTensor + Tensor, float values
    indices8 = np.array([[0, 0, 0], [1, 2, 1]], dtype=np.int64)
    values8 = np.array([1.5, 2.7], dtype=np.float32)
    shape8 = np.array([2, 3, 2], dtype=np.int64)
    a = tf.SparseTensor(indices8, values8, shape8)
    b = np.random.rand(2, 3, 2).astype(np.float32)
    threshold = np.float32(0.0)
    input_dict = {"a": a, "b": b, "threshold": threshold}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tensor + 3D SparseTensor, float values
    a = np.random.rand(2, 3, 2).astype(np.float32)
    indices9 = np.array([[0, 0, 0], [1, 2, 1]], dtype=np.int64)
    values9 = np.array([1.5, 2.7], dtype=np.float32)
    shape9 = np.array([2, 3, 2], dtype=np.int64)
    b = tf.SparseTensor(indices9, values9, shape9)
    threshold = np.float32(0.0)
    input_dict = {"a": a, "b": b, "threshold": threshold}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D SparseTensor + SparseTensor, float values, threshold
    indices10_a = np.array([[0, 0, 0], [1, 2, 1]], dtype=np.int64)
    values10_a = np.array([1.5, 2.7], dtype=np.float32)
    shape10_a = np.array([2, 3, 2], dtype=np.int64)
    a = tf.SparseTensor(indices10_a, values10_a, shape10_a)
    indices10_b = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int64)
    values10_b = np.array([-1.5, 3.0], dtype=np.float32)
    shape10_b = np.array([2, 3, 2], dtype=np.int64)
    b = tf.SparseTensor(indices10_b, values10_b, shape10_b)
    threshold = np.float32(0.1)
    input_dict = {"a": a, "b": b, "threshold": threshold}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.add"] = tf_sparse_add_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.add' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.add'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.sparse.add', generated_inputs['tf.sparse.add'], lib="tf", suffix=0)
