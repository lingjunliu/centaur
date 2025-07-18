
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_deserialize_many_sparse_inputs():
    """
    Generates a list of valid inputs for the tf.io.deserialize_many_sparse function.
    To avoid potential bugs, all SparseTensors within a single batch are created with
    the same dense_shape.
    """
    
    # Helper function to create a batch of serialized sparse tensors
    def create_serialized_batch(sparse_tensors):
        serialized_list = [tf.io.serialize_sparse(st) for st in sparse_tensors]
        return tf.stack(serialized_list).numpy()

    list_of_inputs = []

    # --- Input 1: Rank 1, int32, same shape ---
    st1 = tf.SparseTensor(indices=[[0], [10]], values=[1, 2], dense_shape=[50])
    st2 = tf.SparseTensor(indices=[[2], [20]], values=[4, 5], dense_shape=[50])
    serialized_sparse_1 = create_serialized_batch([st1, st2])
    list_of_inputs.append({
        'serialized_sparse': serialized_sparse_1,
        'dtype': tf.int32,
        'rank': 1,
        'name': 'rank1_same_shape'
    })
    
    # --- Input 2: Rank 2, float32, same shape ---
    st1 = tf.SparseTensor(indices=[[0, 1], [1, 2]], values=[10.1, 20.2], dense_shape=[5, 5])
    st2 = tf.SparseTensor(indices=[[1, 0], [2, 3]], values=[30.3, 40.4], dense_shape=[5, 5])
    st3 = tf.SparseTensor(indices=[[4, 4]], values=[50.5], dense_shape=[5, 5])
    serialized_sparse_2 = create_serialized_batch([st1, st2, st3])
    list_of_inputs.append({
        'serialized_sparse': serialized_sparse_2,
        'dtype': tf.float32,
        'rank': 2,
        'name': 'rank2_same_shape_float32'
    })

    # --- Input 3: Rank 3, int64, same shape ---
    st1 = tf.SparseTensor(indices=[[0, 1, 0]], values=np.array([100], dtype=np.int64), dense_shape=[2, 3, 4])
    st2 = tf.SparseTensor(indices=[[1, 0, 1], [1, 1, 1]], values=np.array([200, 300], dtype=np.int64), dense_shape=[2, 3, 4])
    serialized_sparse_3 = create_serialized_batch([st1, st2])
    list_of_inputs.append({
        'serialized_sparse': serialized_sparse_3,
        'dtype': tf.int64,
        'rank': 3,
        'name': 'rank3_same_shape_int64'
    })

    # --- Input 4: Rank 2, string dtype, same shape ---
    st1 = tf.SparseTensor(indices=[[0, 0], [2, 2]], values=['hello', 'world'], dense_shape=[3, 3])
    st2 = tf.SparseTensor(indices=[[1, 1]], values=['tensorflow'], dense_shape=[3, 3])
    serialized_sparse_4 = create_serialized_batch([st1, st2])
    list_of_inputs.append({
        'serialized_sparse': serialized_sparse_4,
        'dtype': tf.string,
        'rank': 2,
        'name': 'rank2_same_shape_string'
    })

    # --- Input 5: Rank 2, complex64, same shape ---
    st1 = tf.SparseTensor(indices=[[0, 1]], values=np.array([1+2j], dtype=np.complex64), dense_shape=[2, 2])
    st2 = tf.SparseTensor(indices=[[1, 0]], values=np.array([3-4j], dtype=np.complex64), dense_shape=[2, 2])
    serialized_sparse_5 = create_serialized_batch([st1, st2])
    list_of_inputs.append({
        'serialized_sparse': serialized_sparse_5,
        'dtype': tf.complex64,
        'rank': 2,
        'name': 'rank2_same_shape_complex64'
    })

    # --- Input 6: Rank 4, single tensor in batch ---
    st1 = tf.SparseTensor(indices=[[0, 1, 0, 1], [1, 0, 1, 0]], values=[100.1, 200.2], dense_shape=[2, 2, 2, 2])
    serialized_sparse_6 = create_serialized_batch([st1])
    list_of_inputs.append({
        'serialized_sparse': serialized_sparse_6,
        'dtype': tf.float64,
        'rank': 4,
        'name': 'rank4_single_batch'
    })

    # --- Input 7: Rank 2, with an empty sparse tensor, same shape ---
    st1 = tf.SparseTensor(indices=[[0, 0]], values=[1], dense_shape=[2, 2])
    st2 = tf.SparseTensor(indices=np.empty((0, 2), dtype=np.int64), values=[], dense_shape=[2, 2])
    serialized_sparse_7 = create_serialized_batch([st1, st2])
    list_of_inputs.append({
        'serialized_sparse': serialized_sparse_7,
        'dtype': tf.int32,
        'rank': 2,
        'name': 'rank2_same_shape_with_empty'
    })

    # --- Input 8: Rank 0 (scalars) ---
    st1 = tf.SparseTensor(indices=np.empty((1, 0), dtype=np.int64), values=[1.0], dense_shape=np.array([], dtype=np.int64))
    st2 = tf.SparseTensor(indices=np.empty((1, 0), dtype=np.int64), values=[2.0], dense_shape=np.array([], dtype=np.int64))
    serialized_sparse_8 = create_serialized_batch([st1, st2])
    list_of_inputs.append({
        'serialized_sparse': serialized_sparse_8,
        'dtype': tf.float32,
        'rank': 0,
        'name': 'rank0_scalars'
    })

    # --- Input 9: Rank 2, Boolean dtype, same shape ---
    st1 = tf.SparseTensor(indices=[[0, 0]], values=[True], dense_shape=[2, 2])
    st2 = tf.SparseTensor(indices=[[1, 1]], values=[False], dense_shape=[2, 2])
    serialized_sparse_9 = create_serialized_batch([st1, st2])
    list_of_inputs.append({
        'serialized_sparse': serialized_sparse_9,
        'dtype': tf.bool,
        'rank': 2,
        'name': 'rank2_same_shape_bool'
    })

    # --- Input 10: Rank 1, negative values, same shape ---
    st1 = tf.SparseTensor(indices=[[1]], values=[-10], dense_shape=[5])
    st2 = tf.SparseTensor(indices=[[2], [3]], values=[20, -30], dense_shape=[5])
    serialized_sparse_10 = create_serialized_batch([st1, st2])
    list_of_inputs.append({
        'serialized_sparse': serialized_sparse_10,
        'dtype': tf.int32,
        'rank': 1,
        'name': 'rank1_same_shape_negative'
    })

    return [copy.deepcopy(i) for i in list_of_inputs]

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
