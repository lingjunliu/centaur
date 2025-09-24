
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_deserialize_many_sparse_inputs():
    """
    Generates a list of valid inputs for tf.io.deserialize_many_sparse.
    """
    
    def _create_serialized_input(sparse_tensors):
        """Helper to create the [N, 3] serialized string tensor."""
        processed_tensors = []
        for st in sparse_tensors:
            processed_tensors.append(
                tf.SparseTensor(
                    indices=tf.cast(st.indices, tf.int64),
                    values=st.values,
                    dense_shape=tf.cast(st.dense_shape, tf.int64)
                )
            )
        serialized_list = [tf.io.serialize_sparse(st) for st in processed_tensors]
        return tf.stack(serialized_list).numpy()

    list_of_inputs = []

    # Input 1: Basic 1D integer, rank inferred.
    st1_1 = tf.SparseTensor(indices=[[0], [10], [20]], values=[1, 2, 3], dense_shape=[50])
    st1_2 = tf.SparseTensor(indices=[[2], [10]], values=[4, 5], dense_shape=[30])
    input_dict_1 = {
        'serialized_sparse': _create_serialized_input([st1_1, st1_2]),
        'dtype': np.int32,
        'rank': None,
        'name': 'basic_int_deserializer'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 1D float, rank inferred
    st2_1 = tf.SparseTensor(indices=[[1], [3]], values=[1.1, 2.2], dense_shape=[5])
    st2_2 = tf.SparseTensor(indices=[[0], [4]], values=[3.3, 4.4], dense_shape=[6])
    input_dict_2 = {
        'serialized_sparse': _create_serialized_input([st2_1, st2_2]),
        'dtype': np.float32,
        'rank': None,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))
    
    # Input 3: 2D int64, rank inferred
    st3_1 = tf.SparseTensor(indices=[[0, 1], [1, 0]], values=tf.constant([10, 20], dtype=tf.int64), dense_shape=[2, 2])
    st3_2 = tf.SparseTensor(indices=[[0, 0], [2, 1]], values=tf.constant([30, 40], dtype=tf.int64), dense_shape=[3, 3])
    input_dict_3 = {
        'serialized_sparse': _create_serialized_input([st3_1, st3_2]),
        'dtype': np.int64,
        'rank': None,
        'name': 'int64_rank2_deserializer'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 3D int32, rank inferred
    st4_1 = tf.SparseTensor(indices=[[0, 0, 1], [1, 1, 0]], values=[1, 2], dense_shape=[2, 2, 2])
    st4_2 = tf.SparseTensor(indices=[[0, 1, 0]], values=[3], dense_shape=[2, 2, 2])
    input_dict_4 = {
        'serialized_sparse': _create_serialized_input([st4_1, st4_2]),
        'dtype': np.int32,
        'rank': None,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: With an empty SparseTensor, rank inferred
    st5_1 = tf.SparseTensor(indices=[[5]], values=[99], dense_shape=[10])
    st5_2 = tf.SparseTensor(indices=tf.constant([], shape=(0, 1), dtype=tf.int64), values=tf.constant([], dtype=tf.int32), dense_shape=[8])
    st5_3 = tf.SparseTensor(indices=[[1], [2]], values=[11, 22], dense_shape=[12])
    input_dict_5 = {
        'serialized_sparse': _create_serialized_input([st5_1, st5_2, st5_3]),
        'dtype': np.int32,
        'rank': None,
        'name': 'with_empty_deserializer'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Single SparseTensor in minibatch (N=1), rank inferred
    st6_1 = tf.SparseTensor(indices=[[0, 1], [2, 3]], values=[-5, -10], dense_shape=[4, 4])
    input_dict_6 = {
        'serialized_sparse': _create_serialized_input([st6_1]),
        'dtype': np.int32,
        'rank': None,
        'name': 'single_minibatch'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Boolean dtype, rank inferred
    st7_1 = tf.SparseTensor(indices=[[0], [2]], values=[True, False], dense_shape=[4])
    st7_2 = tf.SparseTensor(indices=[[1]], values=[True], dense_shape=[3])
    input_dict_7 = {
        'serialized_sparse': _create_serialized_input([st7_1, st7_2]),
        'dtype': np.bool_,
        'rank': None,
        'name': 'my_bool_deserializer'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Complex dtype, rank inferred
    st8_1 = tf.SparseTensor(indices=[[0]], values=tf.constant([1+2j], dtype=tf.complex64), dense_shape=[2])
    st8_2 = tf.SparseTensor(indices=[[1]], values=tf.constant([3-4j], dtype=tf.complex64), dense_shape=[2])
    input_dict_8 = {
        'serialized_sparse': _create_serialized_input([st8_1, st8_2]),
        'dtype': np.complex64,
        'rank': None,
        'name': 'complex_deserializer'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Unordered indices, rank inferred
    st9_1 = tf.SparseTensor(indices=[[20], [0], [10]], values=[3, 1, 2], dense_shape=[50])
    st9_2 = tf.SparseTensor(indices=[[10], [2]], values=[5, 4], dense_shape=[30])
    input_dict_9 = {
        'serialized_sparse': _create_serialized_input([st9_1, st9_2]),
        'dtype': np.int32,
        'rank': None,
        'name': 'unordered_deserializer'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10: String dtype, rank inferred
    st10_1 = tf.SparseTensor(indices=[[0]], values=[b"hello"], dense_shape=[2])
    st10_2 = tf.SparseTensor(indices=[[1]], values=[b"world"], dense_shape=[2])
    input_dict_10 = {
        'serialized_sparse': _create_serialized_input([st10_1, st10_2]),
        'dtype': np.str_,
        'rank': None,
        'name': 'string_deserializer'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))
    
    # Input 11: All empty sparse tensors, rank inferred
    st11_1 = tf.SparseTensor(indices=tf.constant([], shape=(0, 2), dtype=tf.int64), values=tf.constant([], dtype=tf.int32), dense_shape=[3, 4])
    st11_2 = tf.SparseTensor(indices=tf.constant([], shape=(0, 2), dtype=tf.int64), values=tf.constant([], dtype=tf.int32), dense_shape=[5, 2])
    input_dict_11 = {
        'serialized_sparse': _create_serialized_input([st11_1, st11_2]),
        'dtype': np.int32,
        'rank': None,
        'name': 'all_empty_deserializer'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    return list_of_inputs

generated_inputs["tf.io.deserialize_many_sparse"] = tf_io_deserialize_many_sparse_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.io.deserialize_many_sparse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.deserialize_many_sparse'.")

check_valid('tf.io.deserialize_many_sparse', generated_inputs['tf.io.deserialize_many_sparse'], lib="tf", suffix=0)
