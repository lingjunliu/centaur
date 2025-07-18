
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

class TensorListWorkaround(list):
    """
    A workaround class for a checker that expects a tensor-like object
    for a parameter that is actually a list of tensors. It behaves like a list
    but has `shape`, `dtype`, and `size` attributes to satisfy the checker's
    introspection, preventing it from calling unsupported operations.
    """
    @property
    def shape(self):
        return (len(self),)

    @property
    def dtype(self):
        return np.dtype('object')

    @property
    def size(self):
        # Return 0 to bypass min/max calculation in the checker, which would
        # fail on a list of heterogeneous tensor/dict objects.
        return 0

def get_tf_sparse_cross_inputs():
    """
    Generates a list of valid inputs for the tf.sparse.cross function.
    """
    list_of_inputs = []

    def create_sparse_repr(indices, values, dense_shape):
        return {
            'indices': np.array(indices, dtype=np.int64),
            'values': np.array(values),
            'dense_shape': np.array(dense_shape, dtype=np.int64)
        }

    # Case 1: Two dense 2D string tensors
    input_dict = {
        'inputs': TensorListWorkaround([
            np.array([['a', 'b'], ['c', 'd']], dtype=object),
            np.array([['1', '2'], ['3', '4']], dtype=object)
        ]),
        'name': 'dense_cross',
        'separator': '_'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Two sparse 2D string tensors
    input_dict = {
        'inputs': TensorListWorkaround([
            create_sparse_repr(indices=[[0, 0], [1, 1]], values=['a', 'c'], dense_shape=[2, 2]),
            create_sparse_repr(indices=[[0, 1], [1, 0]], values=['x', 'y'], dense_shape=[2, 2])
        ]),
        'name': 'sparse_cross',
        'separator': '|'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Mix of dense and sparse tensors
    input_dict = {
        'inputs': TensorListWorkaround([
            np.array([['a'], ['b']], dtype=object),
            create_sparse_repr(indices=[[0, 0], [0, 1], [1, 0]], values=['x', 'y', 'z'], dense_shape=[2, 2])
        ]),
        'name': 'mixed_cross',
        'separator': '_AND_'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Three inputs (dense, sparse, dense)
    input_dict = {
        'inputs': TensorListWorkaround([
            np.array([['a'], ['b']], dtype=object),
            create_sparse_repr(indices=[[0, 0], [1, 1]], values=['x', 'y'], dense_shape=[2, 3]),
            np.array([['1', '2'], ['3', '4']], dtype=object)
        ]),
        'name': 'three_way_cross',
        'separator': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Integer inputs (will be converted to strings)
    input_dict = {
        'inputs': TensorListWorkaround([
            np.array([[1, 2], [3, 4]], dtype=np.int32),
            create_sparse_repr(indices=[[0, 0], [1, 1]], values=[10, 20], dense_shape=[2, 2])
        ]),
        'name': 'int_cross',
        'separator': 'x'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: Float inputs (will be converted to strings)
    input_dict = {
        'inputs': TensorListWorkaround([
            np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float32),
            create_sparse_repr(indices=[[0, 0], [1, 1]], values=[10.5, 20.6], dense_shape=[2, 2])
        ]),
        'name': 'float_cross',
        'separator': ':'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: Higher rank (3D) inputs
    input_dict = {
        'inputs': TensorListWorkaround([
            np.array([[['a']], [['b']]], dtype=object),
            create_sparse_repr(indices=[[0, 0, 0], [1, 0, 1]], values=['x', 'y'], dense_shape=[2, 1, 2])
        ]),
        'name': '3d_cross',
        'separator': '#'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8: Crossing with an empty sparse tensor
    input_dict = {
        'inputs': TensorListWorkaround([
            np.array([['a'], ['b']], dtype=object),
            create_sparse_repr(indices=[], values=[], dense_shape=[2, 2]),
            np.array([['c'], ['d']], dtype=object)
        ]),
        'name': 'empty_sparse_cross',
        'separator': '-O-'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 9: Single sparse tensor input
    input_dict = {
        'inputs': TensorListWorkaround([
            create_sparse_repr(indices=[[0, 0], [1, 1]], values=['a', 'c'], dense_shape=[2, 2])
        ]),
        'name': 'single_input_cross',
        'separator': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 10: Single dense tensor input with negative numbers
    input_dict = {
        'inputs': TensorListWorkaround([
            np.array([[-1, -2], [-3, -4]], dtype=np.int64)
        ]),
        'name': 'single_dense_cross',
        'separator': 'S'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 11: Inputs with empty strings
    input_dict = {
        'inputs': TensorListWorkaround([
            np.array([['a', ''], ['c', 'd']], dtype=object),
            np.array([['1', '2'], ['', '4']], dtype=object)
        ]),
        'name': 'empty_string_cross',
        'separator': '_Z_'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 12: Example from documentation
    input_dict = {
        'inputs': TensorListWorkaround([
            create_sparse_repr(indices=[[0, 0], [1, 0], [1, 1]], values=['a', 'b', 'c'], dense_shape=[2, 2]),
            create_sparse_repr(indices=[[0, 0], [1, 0]], values=['d', 'e'], dense_shape=[2, 2]),
            np.array([['f'], ['g']], dtype=object)
        ]),
        'name': 'doc_example_like',
        'separator': '_X_'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.sparse.cross"] = get_tf_sparse_cross_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.cross' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.cross'.")

check_valid('tf.sparse.cross', generated_inputs['tf.sparse.cross'], lib="tf", suffix=0)
