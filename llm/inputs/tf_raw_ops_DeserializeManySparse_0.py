
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def _serialize_sparse(indices, values, shape, dtype):
    """Helper function to serialize a single SparseTensor."""
    # Handle string arrays for tf.constant
    if dtype == tf.string:
        py_values = values
    else:
        # Ensure numpy array has the correct dtype for tf.constant
        py_values = np.array(values, dtype=dtype.as_numpy_dtype)

    return tf.raw_ops.SerializeSparse(
        sparse_indices=tf.constant(indices, dtype=tf.int64),
        sparse_values=tf.constant(py_values, dtype=dtype),
        sparse_shape=tf.constant(shape, dtype=tf.int64)
    )

def tf_raw_ops_deserializemanysparse_inputs():
    list_of_inputs = []

    # --- Input 1: Basic case from documentation (int32) ---
    indices1 = np.array([[0], [10], [20]], dtype=np.int64)
    values1 = [1, 2, 3]
    shape1 = [50]
    indices2 = np.array([[2], [10]], dtype=np.int64)
    values2 = [4, 5]
    shape2 = [30]
    s1 = _serialize_sparse(indices1, values1, shape1, tf.int32)
    s2 = _serialize_sparse(indices2, values2, shape2, tf.int32)
    input_dict = {
        'serialized_sparse': tf.stack([s1, s2]).numpy(),
        'dtype': np.int32,
        'name': 'doc_example_int32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # --- Input 2: 2D tensors with float32 ---
    indices1 = np.array([[0, 1], [1, 0]], dtype=np.int64)
    values1 = [1.1, 2.2]
    shape1 = [2, 2]
    indices2 = np.array([[2, 3]], dtype=np.int64)
    values2 = [3.3]
    shape2 = [4, 5]
    s1 = _serialize_sparse(indices1, values1, shape1, tf.float32)
    s2 = _serialize_sparse(indices2, values2, shape2, tf.float32)
    input_dict = {
        'serialized_sparse': tf.stack([s1, s2]).numpy(),
        'dtype': np.float32,
        'name': '2d_float32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # --- Input 3: 3D tensors with int64 and mismatched shapes ---
    indices1 = np.array([[0, 0, 1], [1, 1, 1]], dtype=np.int64)
    values1 = [10, 20]
    shape1 = [2, 2, 2]
    indices2 = np.array([[0, 1, 0], [2, 0, 1]], dtype=np.int64)
    values2 = [30, 40]
    shape2 = [3, 2, 3]
    s1 = _serialize_sparse(indices1, values1, shape1, tf.int64)
    s2 = _serialize_sparse(indices2, values2, shape2, tf.int64)
    input_dict = {
        'serialized_sparse': tf.stack([s1, s2]).numpy(),
        'dtype': np.int64,
        'name': '3d_int64_mismatched_shape'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # --- Input 4: One empty tensor, float64, negative values ---
    indices1 = np.empty(shape=(0, 2), dtype=np.int64)
    values1 = []
    shape1 = [5, 5]
    indices2 = np.array([[1, 1], [2, 2]], dtype=np.int64)
    values2 = [-1.5, -2.5]
    shape2 = [3, 3]
    s1 = _serialize_sparse(indices1, values1, shape1, tf.float64)
    s2 = _serialize_sparse(indices2, values2, shape2, tf.float64)
    input_dict = {
        'serialized_sparse': tf.stack([s1, s2]).numpy(),
        'dtype': np.float64,
        'name': 'one_empty_float64_neg'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # --- Input 5: All empty tensors ---
    s1 = _serialize_sparse(np.empty((0,3), dtype=np.int64), [], [2,2,2], tf.int32)
    s2 = _serialize_sparse(np.empty((0,3), dtype=np.int64), [], [3,1,4], tf.int32)
    input_dict = {
        'serialized_sparse': tf.stack([s1, s2]).numpy(),
        'dtype': np.int32,
        'name': 'all_empty'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # --- Input 6: Larger minibatch (N=4) ---
    s1 = _serialize_sparse([[0]], [1], [10], tf.int32)
    s2 = _serialize_sparse([[1]], [2], [5], tf.int32)
    s3 = _serialize_sparse([[2]], [3], [12], tf.int32)
    s4 = _serialize_sparse([[3]], [4], [8], tf.int32)
    input_dict = {
        'serialized_sparse': tf.stack([s1, s2, s3, s4]).numpy(),
        'dtype': np.int32,
        'name': 'large_minibatch_n4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # --- Input 7: Complex dtype (complex64) ---
    indices1 = np.array([[0, 0], [1, 1]], dtype=np.int64)
    values1 = [1+2j, 3+4j]
    shape1 = [2, 2]
    indices2 = np.array([[0, 1]], dtype=np.int64)
    values2 = [5-6j]
    shape2 = [1, 3]
    s1 = _serialize_sparse(indices1, values1, shape1, tf.complex64)
    s2 = _serialize_sparse(indices2, values2, shape2, tf.complex64)
    input_dict = {
        'serialized_sparse': tf.stack([s1, s2]).numpy(),
        'dtype': np.complex64,
        'name': 'complex64_values'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # --- Input 8: Minimal case (N=1) ---
    indices1 = np.array([[10], [20]], dtype=np.int64)
    values1 = [-100, -200]
    shape1 = [100]
    s1 = _serialize_sparse(indices1, values1, shape1, tf.int32)
    input_dict = {
        'serialized_sparse': tf.reshape(s1, (1, 3)).numpy(),
        'dtype': np.int32,
        'name': 'minimal_minibatch_n1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # --- Input 9: Boolean dtype ---
    indices1 = np.array([[0], [2]], dtype=np.int64)
    values1 = [True, False]
    shape1 = [5]
    indices2 = np.array([[1]], dtype=np.int64)
    values2 = [True]
    shape2 = [3]
    s1 = _serialize_sparse(indices1, values1, shape1, tf.bool)
    s2 = _serialize_sparse(indices2, values2, shape2, tf.bool)
    input_dict = {
        'serialized_sparse': tf.stack([s1, s2]).numpy(),
        'dtype': np.bool_,
        'name': 'bool_values'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # --- Input 10: String dtype ---
    indices1 = np.array([[0], [1]], dtype=np.int64)
    values1 = ["hello", "world"]
    shape1 = [2]
    indices2 = np.array([[0]], dtype=np.int64)
    values2 = ["tensorflow"]
    shape2 = [3]
    s1 = _serialize_sparse(indices1, values1, shape1, tf.string)
    s2 = _serialize_sparse(indices2, values2, shape2, tf.string)
    input_dict = {
        'serialized_sparse': tf.stack([s1, s2]).numpy(),
        'dtype': np.string_,
        'name': 'string_values'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # --- Input 11: uint8 dtype ---
    indices1 = np.array([[1]], dtype=np.int64)
    values1 = [255]
    shape1 = [2]
    indices2 = np.array([[0]], dtype=np.int64)
    values2 = [128]
    shape2 = [3]
    s1 = _serialize_sparse(indices1, values1, shape1, tf.uint8)
    s2 = _serialize_sparse(indices2, values2, shape2, tf.uint8)
    input_dict = {
        'serialized_sparse': tf.stack([s1, s2]).numpy(),
        'dtype': np.uint8,
        'name': 'uint8_values'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # --- Input 12: Complex128 dtype ---
    indices1 = np.array([[0, 0, 0]], dtype=np.int64)
    values1 = [1.123e10 + 2.456e-10j]
    shape1 = [1, 1, 1]
    s1 = _serialize_sparse(indices1, values1, shape1, tf.complex128)
    s2 = _serialize_sparse(np.empty((0,3), dtype=np.int64), [], [2,2,2], tf.complex128)
    input_dict = {
        'serialized_sparse': tf.stack([s1, s2]).numpy(),
        'dtype': np.complex128,
        'name': 'complex128_values'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.DeserializeManySparse"] = tf_raw_ops_deserializemanysparse_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.DeserializeManySparse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DeserializeManySparse'.")

check_valid('tf.raw_ops.DeserializeManySparse', generated_inputs['tf.raw_ops.DeserializeManySparse'], lib="tf", suffix=0)
