
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_serialize_many_sparse_inputs():
    list_of_inputs = []

    def create_sparse_tensor(indices, values, shape):
        return tf.SparseTensor(indices=indices, values=values, dense_shape=shape)

    # Input 1: Basic case with rank 2
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 2]])
    values = np.array([1, 2, 3, 4])
    shape = np.array([2, 3])
    sp_input = create_sparse_tensor(indices, values, shape)
    input_dict = {"sp_input": sp_input, "out_type": tf.string, "name": "sparse_tensor_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Rank 3 SparseTensor
    indices = np.array([[0, 0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1]])
    values = np.array([1, 2, 3, 4])
    shape = np.array([2, 2, 2])
    sp_input = create_sparse_tensor(indices, values, shape)
    input_dict = {"sp_input": sp_input, "out_type": tf.string, "name": "sparse_tensor_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Empty SparseTensor
    indices = np.array([], dtype=np.int64).reshape(0, 2)
    values = np.array([])
    shape = np.array([2, 3])
    sp_input = create_sparse_tensor(indices, values, shape)
    input_dict = {"sp_input": sp_input, "out_type": tf.string, "name": "sparse_tensor_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Rank 4 SparseTensor
    indices = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]])
    values = np.array([1, 2, 3, 4])
    shape = np.array([2, 2, 2, 2])
    sp_input = create_sparse_tensor(indices, values, shape)
    input_dict = {"sp_input": sp_input, "out_type": tf.string, "name": "sparse_tensor_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger minibatch size
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 2], [2, 1], [2,2]])
    values = np.array([1, 2, 3, 4, 5, 6])
    shape = np.array([3, 3])
    sp_input = create_sparse_tensor(indices, values, shape)
    input_dict = {"sp_input": sp_input, "out_type": tf.string, "name": "sparse_tensor_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: SparseTensor with rank R > 1 and first dim = 1.
    indices = np.array([[0, 0, 0], [0, 0, 1]])
    values = np.array([1, 2])
    shape = np.array([1, 1, 2])
    sp_input = create_sparse_tensor(indices, values, shape)
    input_dict = {"sp_input": sp_input, "out_type": tf.string, "name": "sparse_tensor_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different values
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 2]])
    values = np.array([1.5, 2.5, 3.5, 4.5])
    shape = np.array([2, 3])
    sp_input = create_sparse_tensor(indices, values, shape)
    input_dict = {"sp_input": sp_input, "out_type": tf.string, "name": "sparse_tensor_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: with name = None
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 2]])
    values = np.array([1, 2, 3, 4])
    shape = np.array([2, 3])
    sp_input = create_sparse_tensor(indices, values, shape)
    input_dict = {"sp_input": sp_input, "out_type": tf.string, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: All zeros - removing the shape.size condition - and making values float32
    indices = np.array([[0, 0], [1, 1]])
    values = np.array([0.0, 0.0], dtype=np.float32)
    shape = np.array([2, 2])
    sp_input = create_sparse_tensor(indices, values, shape)
    input_dict = {"sp_input": sp_input, "out_type": tf.string, "name": "sparse_tensor_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Float values
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 2]])
    values = np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float32)
    shape = np.array([2, 3])
    sp_input = create_sparse_tensor(indices, values, shape)
    input_dict = {"sp_input": sp_input, "out_type": tf.string, "name": "sparse_tensor_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.serialize_many_sparse"] = tf_io_serialize_many_sparse_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.serialize_many_sparse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.serialize_many_sparse'.")

check_valid('tf.io.serialize_many_sparse', generated_inputs['tf.io.serialize_many_sparse'], lib="tf", suffix=0)
