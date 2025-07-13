
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_add_inputs():
    list_of_inputs = []

    def get_range(tensor):
        if isinstance(tensor, tf.sparse.SparseTensor):
            return [np.min(tensor.values.numpy()), np.max(tensor.values.numpy())] if tf.size(tensor.values).numpy() > 0 else [0, 0]
        elif isinstance(tensor, tf.Tensor):
            return [np.min(tensor.numpy()), np.max(tensor.numpy())] if tensor.numpy().size > 0 else [0, 0]
        else:
            return [np.min(tensor), np.max(tensor)]

    # Input 1: SparseTensor + Tensor
    a = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[2, 3])
    b = tf.constant([[1, 0, 0], [0, 0, 3]], dtype=tf.int32)
    threshold = tf.constant(0.0, dtype=tf.float32)
    input_dict = {"a": a, "b": b, "threshold": threshold}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Tensor + SparseTensor
    a = tf.constant([[1, 0, 0], [0, 0, 3]], dtype=tf.int32)
    b = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[2, 3])
    threshold = tf.constant(0.0, dtype=tf.float32)
    input_dict = {"a": a, "b": b, "threshold": threshold}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: SparseTensor + SparseTensor, threshold = 0
    a = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[2, 3])
    b = tf.sparse.SparseTensor(indices=[[0, 0], [1, 1]], values=[3, -2], dense_shape=[2, 3])
    threshold = tf.constant(0.0, dtype=tf.float32)
    input_dict = {"a": a, "b": b, "threshold": threshold}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: SparseTensor + SparseTensor, threshold > 0
    a = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[2, 3])
    b = tf.sparse.SparseTensor(indices=[[0, 0], [1, 1]], values=[-1, 2], dense_shape=[2, 3])
    threshold = tf.constant(0.5, dtype=tf.float32)
    input_dict = {"a": a, "b": b, "threshold": threshold}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D SparseTensor + Tensor
    a = tf.sparse.SparseTensor(indices=[[0, 0, 0], [1, 1, 2]], values=[1, 2], dense_shape=[2, 2, 3])
    b = tf.constant([[[1, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 3]]], dtype=tf.int32)
    threshold = tf.constant(0.0, dtype=tf.float32)
    input_dict = {"a": a, "b": b, "threshold": threshold}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensor + 3D SparseTensor
    a = tf.constant([[[1, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 3]]], dtype=tf.int32)
    b = tf.sparse.SparseTensor(indices=[[0, 0, 0], [1, 1, 2]], values=[1, 2], dense_shape=[2, 2, 3])
    threshold = tf.constant(0.0, dtype=tf.float32)
    input_dict = {"a": a, "b": b, "threshold": threshold}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: SparseTensor + SparseTensor, float32 values
    a = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]], values=[1.0, 2.0], dense_shape=[2, 3])
    b = tf.sparse.SparseTensor(indices=[[0, 0], [1, 1]], values=[3.0, -2.0], dense_shape=[2, 3])
    threshold = tf.constant(0.0, dtype=tf.float32)
    input_dict = {"a": a, "b": b, "threshold": threshold}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: SparseTensor + SparseTensor, float32 values, threshold > 0
    a = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]], values=[1.0, 2.0], dense_shape=[2, 3])
    b = tf.sparse.SparseTensor(indices=[[0, 0], [1, 1]], values=[-1.0, 2.0], dense_shape=[2, 3])
    threshold = tf.constant(0.5, dtype=tf.float32)
    input_dict = {"a": a, "b": b, "threshold": threshold}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Complex SparseTensor
    a = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]], values=[1 + 1j, 2 + 2j], dense_shape=[2, 3])
    b = tf.constant([[1, 0, 0], [0, 0, 3]], dtype=tf.complex128)
    threshold = tf.constant(0.0, dtype=tf.float64) # Threshold should be float64 for complex128
    input_dict = {"a": a, "b": b, "threshold": threshold}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: SparseTensor + Tensor, with negative values and float64
    a = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]], values=[-1.5, 2.5], dense_shape=[2, 3])
    b = tf.constant([[1.0, 0.0, 0.0], [0.0, 0.0, -3.0]], dtype=tf.float64)
    threshold = tf.constant(0.0, dtype=tf.float64)
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
