
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_io_serialize_many_sparse_inputs():
    list_of_inputs = []

    # Input 1: Basic valid case
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 2]])
    values = np.array([1, 2, 3, 4])
    shape = np.array([2, 3])
    sp_input = tf.SparseTensor(indices, values, shape)
    out_type = tf.string
    name = "sparse_tensor_1"

    input_dict = {
        "sp_input": sp_input,
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different data type for values
    indices = np.array([[0, 0], [0, 1], [1, 0]])
    values = np.array([1.0, 2.0, 3.0])
    shape = np.array([2, 2])
    sp_input = tf.SparseTensor(indices, values, shape)
    out_type = tf.string
    name = "sparse_tensor_2"

    input_dict = {
        "sp_input": sp_input,
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Empty sparse tensor
    indices = np.array([]).reshape(0, 2)
    values = np.array([])
    shape = np.array([2, 3])
    sp_input = tf.SparseTensor(indices, values, shape)
    out_type = tf.string
    name = "sparse_tensor_3"

    input_dict = {
        "sp_input": sp_input,
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D sparse tensor
    indices = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 2]])
    values = np.array([1, 2, 3])
    shape = np.array([2, 2, 3])
    sp_input = tf.SparseTensor(indices, values, shape)
    out_type = tf.string
    name = "sparse_tensor_4"

    input_dict = {
        "sp_input": sp_input,
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger minibatch size
    indices = np.array([[0, 0], [0, 1], [1, 0], [2, 2], [2, 3]])
    values = np.array([1, 2, 3, 4, 5])
    shape = np.array([3, 4])
    sp_input = tf.SparseTensor(indices, values, shape)
    out_type = tf.string
    name = "sparse_tensor_5"

    input_dict = {
        "sp_input": sp_input,
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Boolean values
    indices = np.array([[0, 0], [0, 1], [1, 0]])
    values = np.array([True, False, True])
    shape = np.array([2, 2])
    sp_input = tf.SparseTensor(indices, values, shape)
    out_type = tf.string
    name = "sparse_tensor_6"

    input_dict = {
        "sp_input": sp_input,
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Int64 indices
    indices = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
    values = np.array([1, 2, 3])
    shape = np.array([2, 2])
    sp_input = tf.SparseTensor(indices, values, shape)
    out_type = tf.string
    name = "sparse_tensor_7"

    input_dict = {
        "sp_input": sp_input,
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8:  4D sparse tensor
    indices = np.array([[0, 0, 0, 0], [0, 1, 1, 1], [1, 0, 2, 2]])
    values = np.array([1, 2, 3])
    shape = np.array([2, 2, 3, 3])
    sp_input = tf.SparseTensor(indices, values, shape)
    out_type = tf.string
    name = "sparse_tensor_8"

    input_dict = {
        "sp_input": sp_input,
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: out_type = tf.string (explicit)
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 2]])
    values = np.array([1, 2, 3, 4])
    shape = np.array([2, 3])
    sp_input = tf.SparseTensor(indices, values, shape)
    out_type = tf.string
    name = "sparse_tensor_9"

    input_dict = {
        "sp_input": sp_input,
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different Values
    indices = np.array([[0, 0], [0, 1], [1, 0]])
    values = np.array([-1, -2, -3])
    shape = np.array([2, 2])
    sp_input = tf.SparseTensor(indices, values, shape)
    out_type = tf.string
    name = "sparse_tensor_10"

    input_dict = {
        "sp_input": sp_input,
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.serialize_many_sparse"] = tf_io_serialize_many_sparse_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.io.serialize_many_sparse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.serialize_many_sparse'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.io.serialize_many_sparse', generated_inputs['tf.io.serialize_many_sparse'], lib="tf", suffix=0)
