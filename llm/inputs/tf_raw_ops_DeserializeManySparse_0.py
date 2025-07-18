
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_deserialize_many_sparse_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.DeserializeManySparse.
    """
    def _serialize_sparse(indices, values, shape, values_dtype_tf):
        """
        Helper to serialize a single sparse tensor.
        """
        rank = len(shape)
        # Ensure indices has correct shape [num_elements, rank], even if empty
        indices_np = np.array(indices, dtype=np.int64).reshape(-1, rank)

        indices_tensor = tf.constant(indices_np, dtype=tf.int64)
        values_tensor = tf.constant(values, dtype=values_dtype_tf)
        shape_tensor = tf.constant(shape, dtype=tf.int64)

        serialized_tensor = tf.raw_ops.SerializeSparse(
            sparse_indices=indices_tensor,
            sparse_values=values_tensor,
            sparse_shape=shape_tensor
        )
        return serialized_tensor.numpy()

    list_of_inputs = []

    # Input 1: From documentation example, int32 type
    dtype_np, dtype_tf = np.int32, tf.int32
    st1 = _serialize_sparse(indices=[[0], [10], [20]], values=[1, 2, 3], shape=[50], values_dtype_tf=dtype_tf)
    st2 = _serialize_sparse(indices=[[2], [10]], values=[4, 5], shape=[30], values_dtype_tf=dtype_tf)
    input_dict = {
        'name': 'doc_example_int32',
        'serialized_sparse': np.stack([st1, st2]),
        'dtype': dtype_np
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D sparse tensors, float32 type
    dtype_np, dtype_tf = np.float32, tf.float32
    st1 = _serialize_sparse(indices=[[0, 0], [1, 1]], values=[1.1, 2.2], shape=[2, 2], values_dtype_tf=dtype_tf)
    st2 = _serialize_sparse(indices=[[0, 1]], values=[3.3], shape=[1, 3], values_dtype_tf=dtype_tf)
    st3 = _serialize_sparse(indices=[[2, 0]], values=[4.4], shape=[3, 1], values_dtype_tf=dtype_tf)
    input_dict = {
        'name': '2d_float32_tensors',
        'serialized_sparse': np.stack([st1, st2, st3]),
        'dtype': dtype_np
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Single 3D sparse tensor, complex64 type
    dtype_np, dtype_tf = np.complex64, tf.complex64
    st1 = _serialize_sparse(indices=[[0, 1, 0], [1, 0, 1]], values=[1+2j, 3-4j], shape=[2, 2, 2], values_dtype_tf=dtype_tf)
    input_dict = {
        'name': 'single_3d_complex64',
        'serialized_sparse': np.expand_dims(st1, axis=0),
        'dtype': dtype_np
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Mix of populated and empty tensors, int64 type
    dtype_np, dtype_tf = np.int64, tf.int64
    st1 = _serialize_sparse(indices=[[5]], values=[100], shape=[10], values_dtype_tf=dtype_tf)
    st2 = _serialize_sparse(indices=[], values=[], shape=[5], values_dtype_tf=dtype_tf)
    st3 = _serialize_sparse(indices=[[0], [8]], values=[200, 300], shape=[9], values_dtype_tf=dtype_tf)
    st4 = _serialize_sparse(indices=[], values=[], shape=[12], values_dtype_tf=dtype_tf)
    input_dict = {
        'name': 'mixed_empty_int64',
        'serialized_sparse': np.stack([st1, st2, st3, st4]),
        'dtype': dtype_np
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Test max shape calculation, float64 type
    dtype_np, dtype_tf = np.float64, tf.float64
    st1 = _serialize_sparse(indices=[[0, 0, 0]], values=[3.14], shape=[1, 1, 1], values_dtype_tf=dtype_tf)
    st2 = _serialize_sparse(indices=[[9, 9, 9]], values=[2.71], shape=[10, 10, 10], values_dtype_tf=dtype_tf)
    input_dict = {
        'name': '3d_float64_max_shape',
        'serialized_sparse': np.stack([st1, st2]),
        'dtype': dtype_np
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Single empty sparse tensor, bool type
    dtype_np, dtype_tf = np.bool_, tf.bool
    st1 = _serialize_sparse(indices=[], values=[], shape=[5, 5], values_dtype_tf=dtype_tf)
    input_dict = {
        'name': 'single_empty_bool',
        'serialized_sparse': np.expand_dims(st1, axis=0),
        'dtype': dtype_np
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensors with negative integer values
    dtype_np, dtype_tf = np.int32, tf.int32
    st1 = _serialize_sparse(indices=[[0], [2]], values=[-10, -20], shape=[3], values_dtype_tf=dtype_tf)
    st2 = _serialize_sparse(indices=[[1]], values=[-30], shape=[4], values_dtype_tf=dtype_tf)
    input_dict = {
        'name': 'negative_int32_values',
        'serialized_sparse': np.stack([st1, st2]),
        'dtype': dtype_np
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger minibatch (N=5), uint8 type
    dtype_np, dtype_tf = np.uint8, tf.uint8
    sts = []
    for i in range(5):
        sts.append(_serialize_sparse(indices=[[i, i]], values=[i], shape=[6, 6], values_dtype_tf=dtype_tf))
    input_dict = {
        'name': 'large_minibatch_uint8',
        'serialized_sparse': np.stack(sts),
        'dtype': dtype_np
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Shape expansion test, float16 type
    dtype_np, dtype_tf = np.float16, tf.float16
    st1 = _serialize_sparse(indices=[[0, 0]], values=[1.0], shape=[2, 2], values_dtype_tf=dtype_tf)
    st2 = _serialize_sparse(indices=[[0, 0]], values=[2.0], shape=[1, 100], values_dtype_tf=dtype_tf)
    input_dict = {
        'name': 'shape_expansion_float16',
        'serialized_sparse': np.stack([st1, st2]),
        'dtype': dtype_np
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: String values
    dtype_np, dtype_tf = np.object_, tf.string
    st1 = _serialize_sparse(indices=[[0]], values=[b"hello"], shape=[5], values_dtype_tf=dtype_tf)
    st2 = _serialize_sparse(indices=[[1], [3]], values=[b"world", b"!"], shape=[5], values_dtype_tf=dtype_tf)
    input_dict = {
        'name': 'string_values',
        'serialized_sparse': np.stack([st1, st2]),
        'dtype': dtype_np
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: int16 tensors with mix of empty and non-empty
    dtype_np, dtype_tf = np.int16, tf.int16
    st1 = _serialize_sparse(indices=[[0, 1], [1, 0]], values=[10, 20], shape=[2, 2], values_dtype_tf=dtype_tf)
    st2 = _serialize_sparse(indices=[[2, 2]], values=[30], shape=[3, 3], values_dtype_tf=dtype_tf)
    st3 = _serialize_sparse(indices=[], values=[], shape=[1, 1], values_dtype_tf=dtype_tf)
    input_dict = {
        'name': 'mixed_int16',
        'serialized_sparse': np.stack([st1, st2, st3]),
        'dtype': dtype_np
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.DeserializeManySparse"] = tf_raw_ops_deserialize_many_sparse_inputs()

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
