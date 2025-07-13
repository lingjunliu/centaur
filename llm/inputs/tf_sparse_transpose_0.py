
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_transpose_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D SparseTensor transpose
    indices = np.array([[0, 1], [1, 0], [2, 2]])
    values = np.array([1, 2, 3])
    dense_shape = np.array([3, 3])
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    perm = None
    name = "transpose_1"
    input_dict = {"sp_input": sp_input, "perm": perm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D SparseTensor with specific permutation
    indices = np.array([[0, 1], [1, 0], [2, 2]])
    values = np.array([1, 2, 3])
    dense_shape = np.array([3, 3])
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    perm = [1, 0]
    name = "transpose_2"
    input_dict = {"sp_input": sp_input, "perm": perm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D SparseTensor transpose with default permutation
    indices = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    values = np.array([1, 2, 3])
    dense_shape = np.array([2, 2, 2])
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    perm = None
    name = "transpose_3"
    input_dict = {"sp_input": sp_input, "perm": perm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D SparseTensor transpose with specific permutation
    indices = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    values = np.array([1, 2, 3])
    dense_shape = np.array([2, 2, 2])
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    perm = [0, 2, 1]
    name = "transpose_4"
    input_dict = {"sp_input": sp_input, "perm": perm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger 2D SparseTensor
    indices = np.array([[0, 2], [1, 4], [3, 0], [4, 1]])
    values = np.array([1, 2, 3, 4])
    dense_shape = np.array([5, 5])
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    perm = None
    name = "transpose_5"
    input_dict = {"sp_input": sp_input, "perm": perm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6: Empty SparseTensor
    indices = np.array([]).reshape(0, 2)
    values = np.array([])
    dense_shape = np.array([2, 2])
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    perm = None
    name = "transpose_6"
    input_dict = {"sp_input": sp_input, "perm": perm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Sparse Tensor with duplicate indices (valid as per TF documentation)
    indices = np.array([[0, 1], [0, 1], [1, 0]])
    values = np.array([1, 2, 3])
    dense_shape = np.array([2, 2])
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    perm = None
    name = "transpose_7"
    input_dict = {"sp_input": sp_input, "perm": perm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger 3D SparseTensor
    indices = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    values = np.array([1, 2, 3, 4, 5])
    dense_shape = np.array([2, 2, 2])
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    perm = [2, 0, 1]
    name = "transpose_8"
    input_dict = {"sp_input": sp_input, "perm": perm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D SparseTensor with permutation - Reduced to avoid size issues
    indices = np.array([[0, 0, 1, 0], [0, 1, 0, 0]])
    values = np.array([1, 2])
    dense_shape = np.array([2, 2, 2, 2])
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    perm = [3, 1, 0, 2]
    name = "transpose_9"
    input_dict = {"sp_input": sp_input, "perm": perm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D SparseTensor with a name that includes special characters
    indices = np.array([[0, 1], [1, 0], [2, 2]])
    values = np.array([1, 2, 3])
    dense_shape = np.array([3, 3])
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    perm = [1, 0]
    name = "transpose-10_special.chars"
    input_dict = {"sp_input": sp_input, "perm": perm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.transpose"] = tf_sparse_transpose_inputs()

def check_valid(api, inputs, lib="tf", suffix=0):
    from inspect import signature

    def get_signature(api, lib="tf", suffix=0):
        sig = signature(eval(api))
        parameters = []
        args = []
        for param in sig.parameters.values():
            parameters.append(str(param).split(':')[0])
        return parameters

    def get_abstract_input(input_dict, signature):
        abstract = {}
        for arg in signature:
            value = input_dict[arg]

            def get_ll(domain, value):
                if isinstance(value, str):
                    return ['"' + value + '"', '"string"']
                if isinstance(value, bool):
                    return [str(value).lower(), "bool"]
                if isinstance(value, int):
                    return [str(value), "int"]
                if isinstance(value, float):
                    return [str(value), "float"]
                if isinstance(value, np.ndarray):
                    return [str(value.tolist()), f"ndarray"]
                if isinstance(value, list):
                    return [str(value), "list"]
                if isinstance(value, tf.SparseTensor):
                    return ["tf.SparseTensor", "SparseTensor"]
                try:
                    if hasattr(value, 'numpy'):
                        value = value.numpy()
                        return [str(value.tolist()), f"ndarray"]
                except:
                  pass
                return ["None", "NoneType"]

            [val, domain] = get_ll("", value)
            abstract[arg] = domain
        return abstract

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sparse.transpose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.transpose'.")

check_valid('tf.sparse.transpose', generated_inputs['tf.sparse.transpose'], lib="tf", suffix=0)
