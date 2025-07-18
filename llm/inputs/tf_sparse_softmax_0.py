
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

class NumpyCompliantSparseTensor:
    """
    A proxy object that is compatible with numpy-based analysis tools
    (via .shape, .size, and __array__) but converts to a tf.SparseTensor
    when passed to a TensorFlow function.
    """
    def __init__(self, indices, values, dense_shape):
        self.indices = np.array(indices, dtype=np.int64)
        self.values = np.array(values)
        self.dense_shape = np.array(dense_shape, dtype=np.int64)

    @property
    def shape(self):
        """The shape of the dense version of the sparse tensor."""
        return self.dense_shape

    @property
    def size(self):
        """The number of non-zero elements."""
        return self.values.size

    def __array__(self):
        """Allows numpy functions to operate on the non-zero values."""
        return self.values

    def to_tf_sparse_tensor(self):
        """Converts this object to an actual tf.SparseTensor."""
        return tf.SparseTensor(self.indices, self.values, self.dense_shape)

# Register the conversion function so tf.convert_to_tensor understands this class
def _my_sparse_tensor_converter(value, dtype=None, name=None, as_ref=False):
    return value.to_tf_sparse_tensor()

try:
    tf.register_tensor_conversion_function(
        NumpyCompliantSparseTensor, _my_sparse_tensor_converter, 100)
except ValueError:
    # Conversion function may already be registered if the script is run multiple times.
    pass

def tf_sparse_softmax_inputs():
    """
    Generates a list of valid inputs for the tf.sparse.softmax function
    using a custom proxy class to satisfy both analysis and execution.
    """
    list_of_inputs = []

    # Input 1: From documentation example (3D)
    indices = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]]
    values = [np.e, 1., np.e, np.e, np.e]
    dense_shape = [2, 2, 2]
    list_of_inputs.append(copy.deepcopy({
        'sp_input': NumpyCompliantSparseTensor(indices, values, dense_shape),
        'name': 'doc_example'
    }))

    # Input 2: Basic 2D case
    indices = [[0, 1], [0, 3], [1, 0], [1, 2], [1, 4], [2, 2]]
    values = [1., 2., 3., 4., 5., 6.]
    dense_shape = [3, 5]
    list_of_inputs.append(copy.deepcopy({
        'sp_input': NumpyCompliantSparseTensor(indices, values, dense_shape),
        'name': 'basic_2d'
    }))

    # Input 3: 2D with negative values
    indices = [[0, 0], [0, 2], [1, 1], [1, 3]]
    values = [-1.0, 1.0, -2.0, 0.0]
    dense_shape = [2, 4]
    list_of_inputs.append(copy.deepcopy({
        'sp_input': NumpyCompliantSparseTensor(indices, values, dense_shape),
        'name': 'negative_values_2d'
    }))

    # Input 4: Higher rank (4D)
    indices = [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 2],
               [1, 0, 1, 1], [1, 1, 1, 0], [1, 1, 1, 2]]
    values = [1., 2., 3., 4., 5., 6.]
    dense_shape = [2, 2, 2, 3]
    list_of_inputs.append(copy.deepcopy({
        'sp_input': NumpyCompliantSparseTensor(indices, values, dense_shape),
        'name': 'rank_4'
    }))

    # Input 5: float64 dtype
    indices = [[0, 0], [0, 1], [1, 2], [1, 3]]
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    dense_shape = [2, 4]
    list_of_inputs.append(copy.deepcopy({
        'sp_input': NumpyCompliantSparseTensor(indices, values, dense_shape),
        'name': 'float64_dtype'
    }))

    # Input 6: A batch dimension is empty (implicitly zero)
    indices = [[0, 0, 1], [0, 1, 0], [0, 1, 2]]
    values = [1., 2., 3.]
    dense_shape = [2, 2, 3]
    list_of_inputs.append(copy.deepcopy({
        'sp_input': NumpyCompliantSparseTensor(indices, values, dense_shape),
        'name': 'empty_submatrix'
    }))

    # Input 7: Very sparse tensor
    indices = [[0, 5, 10], [5, 0, 15], [9, 9, 19]]
    values = [1., 2., 3.]
    dense_shape = [10, 10, 20]
    list_of_inputs.append(copy.deepcopy({
        'sp_input': NumpyCompliantSparseTensor(indices, values, dense_shape),
        'name': 'very_sparse'
    }))

    # Input 8: Row with identical values
    indices = [[0, 0], [0, 1], [0, 2], [1, 1], [1, 3]]
    values = [np.log(2.), np.log(2.), np.log(2.), 5., 1.]
    dense_shape = [2, 4]
    list_of_inputs.append(copy.deepcopy({
        'sp_input': NumpyCompliantSparseTensor(indices, values, dense_shape),
        'name': 'same_value_row'
    }))

    # Input 9: Large values
    indices = [[0, 0], [0, 4], [1, 1], [1, 2]]
    values = [1000., 1001., -1000., -1001.]
    dense_shape = [2, 5]
    list_of_inputs.append(copy.deepcopy({
        'sp_input': NumpyCompliantSparseTensor(indices, values, dense_shape),
        'name': 'large_values'
    }))

    # Input 10: 2D tensor with an empty row
    indices = [[0, 1], [0, 3], [2, 0], [2, 4], [3, 1], [3, 2], [3, 3]]
    values = [1., 1., 1., 1., 1., 1., 1.]
    dense_shape = [4, 5]
    list_of_inputs.append(copy.deepcopy({
        'sp_input': NumpyCompliantSparseTensor(indices, values, dense_shape),
        'name': 'empty_row_2d'
    }))

    return list_of_inputs

generated_inputs["tf.sparse.softmax"] = tf_sparse_softmax_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.softmax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.softmax'.")

check_valid('tf.sparse.softmax', generated_inputs['tf.sparse.softmax'], lib="tf", suffix=0)
