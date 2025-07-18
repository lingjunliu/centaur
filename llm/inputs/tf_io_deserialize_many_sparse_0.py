
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
    list_of_inputs = []

    # Input 1: Basic case from doc, rank 2, int32
    st1_1 = tf.SparseTensor(indices=[[0, 1], [1, 2]], values=[1, 2], dense_shape=[3, 4])
    st1_2 = tf.SparseTensor(indices=[[0, 0], [2, 1]], values=[3, 4], dense_shape=[4, 3])
    ser1 = tf.stack([tf.io.serialize_sparse(st1_1), tf.io.serialize_sparse(st1_2)]).numpy()
    list_of_inputs.append({
        'serialized_sparse': ser1,
        'dtype': np.int32,
        'rank': 2,
        'name': 'doc_example_rank2_int32'
    })

    # Input 2: Basic case, rank 2, float32
    st2_1 = tf.SparseTensor(indices=[[1, 3]], values=[1.1], dense_shape=[10, 5])
    st2_2 = tf.SparseTensor(indices=[[0, 4], [8, 1]], values=[3.3, 4.4], dense_shape=[12, 6])
    ser2 = tf.stack([tf.io.serialize_sparse(st2_1), tf.io.serialize_sparse(st2_2)]).numpy()
    list_of_inputs.append({
        'serialized_sparse': ser2,
        'dtype': np.float32,
        'rank': 2,
        'name': 'basic_rank2_float32'
    })

    # Input 3: Basic case, rank 3, int64
    st3_1 = tf.SparseTensor(indices=[[0, 1, 0], [2, 3, 1]], values=np.array([10, 20], dtype=np.int64), dense_shape=[5, 5, 2])
    st3_2 = tf.SparseTensor(indices=[[1, 1, 1], [3, 0, 0], [4, 4, 1]], values=np.array([30, 40, 50], dtype=np.int64), dense_shape=[6, 5, 2])
    ser3 = tf.stack([tf.io.serialize_sparse(st3_1), tf.io.serialize_sparse(st3_2)]).numpy()
    list_of_inputs.append({
        'serialized_sparse': ser3,
        'dtype': np.int64,
        'rank': 3,
        'name': 'basic_rank3_int64'
    })

    # Input 4: Basic case, rank 4, float64
    st4_1 = tf.SparseTensor(indices=[[0, 0, 1, 0], [1, 1, 1, 1]], values=[100.1, 200.2], dense_shape=[2, 2, 2, 2])
    st4_2 = tf.SparseTensor(indices=[[0, 1, 0, 1]], values=[300.3], dense_shape=[2, 3, 2, 2])
    ser4 = tf.stack([tf.io.serialize_sparse(st4_1), tf.io.serialize_sparse(st4_2)]).numpy()
    list_of_inputs.append({
        'serialized_sparse': ser4,
        'dtype': np.float64,
        'rank': 4,
        'name': 'basic_rank4_float64'
    })

    # Input 5: One empty tensor, rank 2
    st5_1 = tf.SparseTensor(indices=[[0,0]], values=[1], dense_shape=[2,2])
    st5_2 = tf.SparseTensor(indices=np.empty((0,2), dtype=np.int64), values=[], dense_shape=[3,3])
    st5_3 = tf.SparseTensor(indices=[[1,1],[2,0]], values=[2,3], dense_shape=[3,2])
    ser5 = tf.stack([tf.io.serialize_sparse(st5_1), tf.io.serialize_sparse(st5_2), tf.io.serialize_sparse(st5_3)]).numpy()
    list_of_inputs.append({
        'serialized_sparse': ser5,
        'dtype': np.int32,
        'rank': 2,
        'name': 'one_empty_rank2'
    })

    # Input 6: All empty tensors, rank 3
    st6_1 = tf.SparseTensor(indices=np.empty((0,3), dtype=np.int64), values=np.array([], dtype=np.float32), dense_shape=[10, 5, 2])
    st6_2 = tf.SparseTensor(indices=np.empty((0,3), dtype=np.int64), values=np.array([], dtype=np.float32), dense_shape=[5, 10, 3])
    ser6 = tf.stack([tf.io.serialize_sparse(st6_1), tf.io.serialize_sparse(st6_2)]).numpy()
    list_of_inputs.append({
        'serialized_sparse': ser6,
        'dtype': np.float32,
        'rank': 3,
        'name': 'all_empty_rank3'
    })

    # Input 7: Single item in batch, rank 2
    st7_1 = tf.SparseTensor(indices=[[0,1],[1,0]], values=[10,20], dense_shape=[2,2])
    ser7 = tf.stack([tf.io.serialize_sparse(st7_1)]).numpy()
    list_of_inputs.append({
        'serialized_sparse': ser7,
        'dtype': np.int32,
        'rank': 2,
        'name': 'single_item_rank2'
    })

    # Input 8: Larger batch, rank 2
    st8_1 = tf.SparseTensor([[0,0]], [1], [2,2])
    st8_2 = tf.SparseTensor([[1,1]], [2], [2,2])
    st8_3 = tf.SparseTensor([[0,1]], [3], [2,2])
    st8_4 = tf.SparseTensor([[1,0]], [4], [2,2])
    ser8 = tf.stack([tf.io.serialize_sparse(st8_1), tf.io.serialize_sparse(st8_2), tf.io.serialize_sparse(st8_3), tf.io.serialize_sparse(st8_4)]).numpy()
    list_of_inputs.append({
        'serialized_sparse': ser8,
        'dtype': np.int32,
        'rank': 2,
        'name': 'large_batch_rank2'
    })
    
    # Input 9: Rank 1 sparse tensors
    st9_1 = tf.SparseTensor(indices=[[0], [5]], values=[10.0, 20.0], dense_shape=[10])
    st9_2 = tf.SparseTensor(indices=[[2]], values=[30.0], dense_shape=[8])
    ser9 = tf.stack([tf.io.serialize_sparse(st9_1), tf.io.serialize_sparse(st9_2)]).numpy()
    list_of_inputs.append({
        'serialized_sparse': ser9,
        'dtype': np.float32,
        'rank': 1,
        'name': 'basic_rank1_float32'
    })
    
    # Input 10: Rank 2, boolean type
    st10_1 = tf.SparseTensor(indices=[[0,0], [1,1]], values=[True, False], dense_shape=[2,2])
    st10_2 = tf.SparseTensor(indices=[[0,1]], values=[True], dense_shape=[2,2])
    ser10 = tf.stack([tf.io.serialize_sparse(st10_1), tf.io.serialize_sparse(st10_2)]).numpy()
    list_of_inputs.append({
        'serialized_sparse': ser10,
        'dtype': np.bool_,
        'rank': 2,
        'name': 'rank2_bool'
    })

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
