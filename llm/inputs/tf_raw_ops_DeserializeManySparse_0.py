
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_tf_raw_ops_deserializemanysparse_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.DeserializeManySparse.
    """
    list_of_inputs = []

    # Helper function to serialize a single sparse tensor.
    def serialize_sparse(indices, values, shape, tf_dtype):
        indices_tf = tf.constant(indices, dtype=tf.int64)
        values_tf = tf.constant(values, dtype=tf_dtype)
        shape_tf = tf.constant(shape, dtype=tf.int64)
        
        # The 'out_type' for SerializeSparse must be string or variant.
        # The actual data type is encoded within the serialization.
        serialized = tf.raw_ops.SerializeSparse(
            sparse_indices=indices_tf,
            sparse_values=values_tf,
            sparse_shape=shape_tf
        )
        return tf.reshape(serialized, (1, 3))

    # Case 1: Doc example (float32, rank 1, N=2)
    st1_indices = np.array([[0], [10], [20]], dtype=np.int64)
    st1_values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    st1_shape = np.array([50], dtype=np.int64)
    st2_indices = np.array([[2], [10]], dtype=np.int64)
    st2_values = np.array([4.0, 5.0], dtype=np.float32)
    st2_shape = np.array([30], dtype=np.int64)
    s1 = serialize_sparse(st1_indices, st1_values, st1_shape, tf.float32)
    s2 = serialize_sparse(st2_indices, st2_values, st2_shape, tf.float32)
    input_dict = {
        'serialized_sparse': tf.concat([s1, s2], axis=0).numpy(),
        'dtype': np.float32,
        'name': 'doc_example_float32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Rank 2, int32, negative values
    st1_indices = np.array([[0, 1], [1, 0]], dtype=np.int64)
    st1_values = np.array([-10, 20], dtype=np.int32)
    st1_shape = np.array([2, 2], dtype=np.int64)
    st2_indices = np.array([[1, 1], [2, 3]], dtype=np.int64)
    st2_values = np.array([30, -40], dtype=np.int32)
    st2_shape = np.array([3, 4], dtype=np.int64)
    s1 = serialize_sparse(st1_indices, st1_values, st1_shape, tf.int32)
    s2 = serialize_sparse(st2_indices, st2_values, st2_shape, tf.int32)
    input_dict = {
        'serialized_sparse': tf.concat([s1, s2], axis=0).numpy(),
        'dtype': np.int32,
        'name': 'rank2_int32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Rank 3, float64, N=3
    st1_indices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int64)
    st1_values = np.array([1.1, 2.2], dtype=np.float64)
    st1_shape = np.array([2, 2, 2], dtype=np.int64)
    st2_indices = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int64)
    st2_values = np.array([3.3, 4.4], dtype=np.float64)
    st2_shape = np.array([2, 2, 2], dtype=np.int64)
    st3_indices = np.array([[0, 0, 1]], dtype=np.int64)
    st3_values = np.array([5.5], dtype=np.float64)
    st3_shape = np.array([2, 2, 2], dtype=np.int64)
    s1 = serialize_sparse(st1_indices, st1_values, st1_shape, tf.float64)
    s2 = serialize_sparse(st2_indices, st2_values, st2_shape, tf.float64)
    s3 = serialize_sparse(st3_indices, st3_values, st3_shape, tf.float64)
    input_dict = {
        'serialized_sparse': tf.concat([s1, s2, s3], axis=0).numpy(),
        'dtype': np.float64,
        'name': 'rank3_float64_n3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: N=1, complex64
    st1_indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    st1_values = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
    st1_shape = np.array([5, 5], dtype=np.int64)
    s1 = serialize_sparse(st1_indices, st1_values, st1_shape, tf.complex64)
    input_dict = {
        'serialized_sparse': s1.numpy(),
        'dtype': np.complex64,
        'name': 'single_tensor_complex64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: One empty tensor, one non-empty, int64
    st1_indices = np.empty((0, 2), dtype=np.int64)
    st1_values = np.array([], dtype=np.int64)
    st1_shape = np.array([10, 10], dtype=np.int64)
    st2_indices = np.array([[1, 1], [2, 2]], dtype=np.int64)
    st2_values = np.array([100, 200], dtype=np.int64)
    st2_shape = np.array([5, 5], dtype=np.int64)
    s1 = serialize_sparse(st1_indices, st1_values, st1_shape, tf.int64)
    s2 = serialize_sparse(st2_indices, st2_values, st2_shape, tf.int64)
    input_dict = {
        'serialized_sparse': tf.concat([s1, s2], axis=0).numpy(),
        'dtype': np.int64,
        'name': 'one_empty_int64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: String values
    st1_indices = np.array([[0], [1]], dtype=np.int64)
    st1_values = np.array([b'hello', b'world'], dtype=object)
    st1_shape = np.array([2], dtype=np.int64)
    st2_indices = np.array([[1]], dtype=np.int64)
    st2_values = np.array([b'tensorflow'], dtype=object)
    st2_shape = np.array([3], dtype=np.int64)
    s1 = serialize_sparse(st1_indices, st1_values, st1_shape, tf.string)
    s2 = serialize_sparse(st2_indices, st2_values, st2_shape, tf.string)
    input_dict = {
        'serialized_sparse': tf.concat([s1, s2], axis=0).numpy(),
        'dtype': tf.string.as_numpy_dtype,
        'name': 'string_values'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: N=4, simple rank 1 tensors
    s_list = []
    for i in range(4):
        indices = np.array([[i], [i + 10]], dtype=np.int64)
        values = np.array([float(i * 10), float(i * 10 + 1)], dtype=np.float32)
        shape = np.array([20], dtype=np.int64)
        s_list.append(serialize_sparse(indices, values, shape, tf.float32))
    input_dict = {
        'serialized_sparse': tf.concat(s_list, axis=0).numpy(),
        'dtype': np.float32,
        'name': 'large_n'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8: Bool dtype
    st1_indices = np.array([[0, 0]], dtype=np.int64)
    st1_values = np.array([True], dtype=np.bool_)
    st1_shape = np.array([1, 1], dtype=np.int64)
    st2_indices = np.array([[1, 0]], dtype=np.int64)
    st2_values = np.array([False], dtype=np.bool_)
    st2_shape = np.array([2, 1], dtype=np.int64)
    s1 = serialize_sparse(st1_indices, st1_values, st1_shape, tf.bool)
    s2 = serialize_sparse(st2_indices, st2_values, st2_shape, tf.bool)
    input_dict = {
        'serialized_sparse': tf.concat([s1, s2], axis=0).numpy(),
        'dtype': np.bool_,
        'name': 'bool_values'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 9: All empty sparse tensors
    st1_indices = np.empty((0, 3), dtype=np.int64)
    st1_values = np.array([], dtype=np.float32)
    st1_shape = np.array([1, 1, 1], dtype=np.int64)
    st2_indices = np.empty((0, 3), dtype=np.int64)
    st2_values = np.array([], dtype=np.float32)
    st2_shape = np.array([2, 2, 2], dtype=np.int64)
    s1 = serialize_sparse(st1_indices, st1_values, st1_shape, tf.float32)
    s2 = serialize_sparse(st2_indices, st2_values, st2_shape, tf.float32)
    input_dict = {
        'serialized_sparse': tf.concat([s1, s2], axis=0).numpy(),
        'dtype': np.float32,
        'name': 'all_empty'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 10: complex128 dtype
    st1_indices = np.array([[0]], dtype=np.int64)
    st1_values = np.array([1.23 + 4.56j], dtype=np.complex128)
    st1_shape = np.array([1], dtype=np.int64)
    st2_indices = np.array([[10]], dtype=np.int64)
    st2_values = np.array([-7.89 - 0.12j], dtype=np.complex128)
    st2_shape = np.array([20], dtype=np.int64)
    s1 = serialize_sparse(st1_indices, st1_values, st1_shape, tf.complex128)
    s2 = serialize_sparse(st2_indices, st2_values, st2_shape, tf.complex128)
    input_dict = {
        'serialized_sparse': tf.concat([s1, s2], axis=0).numpy(),
        'dtype': np.complex128,
        'name': 'complex128_dtype'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 11: uint8 dtype
    st1_indices = np.array([[0]], dtype=np.int64)
    st1_values = np.array([255], dtype=np.uint8)
    st1_shape = np.array([1], dtype=np.int64)
    st2_indices = np.array([[0]], dtype=np.int64)
    st2_values = np.array([0], dtype=np.uint8)
    st2_shape = np.array([1], dtype=np.int64)
    s1 = serialize_sparse(st1_indices, st1_values, st1_shape, tf.uint8)
    s2 = serialize_sparse(st2_indices, st2_values, st2_shape, tf.uint8)
    input_dict = {
        'serialized_sparse': tf.concat([s1, s2], axis=0).numpy(),
        'dtype': np.uint8,
        'name': 'uint8_dtype'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.DeserializeManySparse"] = get_tf_raw_ops_deserializemanysparse_inputs()

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
