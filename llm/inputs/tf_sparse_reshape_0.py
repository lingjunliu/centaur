
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

# Custom class to satisfy the testing framework's requirement for a .size attribute
# while still being a valid SparseTensor for the TensorFlow API.
class CustomSparseTensor(tf.SparseTensor):
    """
    A subclass of tf.SparseTensor that includes a `.size` attribute
    to be compatible with a testing framework that requires it.
    The .size is defined as the number of non-zero elements.
    """
    def __init__(self, indices, values, dense_shape):
        # Convert inputs to Tensors before passing to the parent constructor
        indices_tensor = tf.convert_to_tensor(indices, dtype=tf.int64)
        values_tensor = tf.convert_to_tensor(values)
        dense_shape_tensor = tf.convert_to_tensor(dense_shape, dtype=tf.int64)

        super().__init__(indices_tensor, values_tensor, dense_shape_tensor)
        # The .size is the number of specified values (non-zero elements)
        self.size = tf.size(self.values).numpy()

def tf_sparse_reshape_inputs():
    """
    Generates a list of valid inputs for tf.sparse.reshape.
    """
    list_of_inputs = []

    # Input 1: Basic 2D to 2D reshape
    sp_input_1 = CustomSparseTensor(
        indices=[[0, 1], [1, 4]],
        values=np.array([10, 20], dtype=np.int32),
        dense_shape=[2, 6]
    )
    list_of_inputs.append(copy.deepcopy({
        'sp_input': sp_input_1,
        'shape': np.array([3, 4], dtype=np.int64),
        'name': 'basic_reshape'
    }))

    # Input 2: Flattening a 3D tensor to 1D
    sp_input_2 = CustomSparseTensor(
        indices=[[0, 1, 1], [1, 0, 2]],
        values=np.array([1.1, 2.2], dtype=np.float32),
        dense_shape=[2, 2, 3]
    )
    list_of_inputs.append(copy.deepcopy({
        'sp_input': sp_input_2,
        'shape': np.array([12], dtype=np.int64),
        'name': 'flatten_3d'
    }))

    # Input 3: Using -1 to infer a dimension
    sp_input_3 = CustomSparseTensor(
        indices=[[0, 0], [1, 1], [2, 2], [3, 3]],
        values=np.array([1, 2, 3, 4], dtype=np.int32),
        dense_shape=[4, 6]
    )
    list_of_inputs.append(copy.deepcopy({
        'sp_input': sp_input_3,
        'shape': np.array([8, -1], dtype=np.int64)
    }))

    # Input 4: Reshaping from 3D to 2D
    sp_input_4 = CustomSparseTensor(
        indices=[[0, 1, 2], [1, 2, 0]],
        values=np.array([100, 200], dtype=np.int64),
        dense_shape=[2, 3, 4]
    )
    list_of_inputs.append(copy.deepcopy({
        'sp_input': sp_input_4,
        'shape': np.array([6, 4], dtype=np.int64)
    }))

    # Input 5: Reshaping an empty SparseTensor
    sp_input_5 = CustomSparseTensor(
        indices=np.empty((0, 2), dtype=np.int64),
        values=np.array([], dtype=np.float32),
        dense_shape=[5, 10]
    )
    list_of_inputs.append(copy.deepcopy({
        'sp_input': sp_input_5,
        'shape': np.array([2, 25], dtype=np.int64),
        'name': 'reshape_empty'
    }))

    # Input 6: Reshaping a full SparseTensor
    sp_input_6 = CustomSparseTensor(
        indices=[[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]],
        values=np.array([1, 2, 3, 4, 5, 6], dtype=np.int32),
        dense_shape=[2, 3]
    )
    list_of_inputs.append(copy.deepcopy({
        'sp_input': sp_input_6,
        'shape': np.array([3, 2], dtype=np.int64)
    }))

    # Input 7: Using float64 values
    sp_input_7 = CustomSparseTensor(
        indices=[[1, 1], [3, 3]],
        values=np.array([3.14, 2.71], dtype=np.float64),
        dense_shape=[4, 4]
    )
    list_of_inputs.append(copy.deepcopy({
        'sp_input': sp_input_7,
        'shape': np.array([2, 8], dtype=np.int64)
    }))

    # Input 8: Reshaping to a single column vector
    sp_input_8 = CustomSparseTensor(
        indices=[[0, 7], [1, 0], [2, 3]],
        values=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        dense_shape=[3, 8]
    )
    list_of_inputs.append(copy.deepcopy({
        'sp_input': sp_input_8,
        'shape': np.array([-1, 1], dtype=np.int64)
    }))

    # Input 9: Identity reshape with negative values
    sp_input_9 = CustomSparseTensor(
        indices=[[0, 1], [1, 3], [2, 0]],
        values=np.array([-1, -2, -3], dtype=np.int32),
        dense_shape=[3, 5]
    )
    list_of_inputs.append(copy.deepcopy({
        'sp_input': sp_input_9,
        'shape': np.array([3, 5], dtype=np.int64),
        'name': 'identity_reshape'
    }))

    # Input 10: Boolean values
    sp_input_10 = CustomSparseTensor(
        indices=[[0]],
        values=np.array([True], dtype=np.bool_),
        dense_shape=[1]
    )
    list_of_inputs.append(copy.deepcopy({
        'sp_input': sp_input_10,
        'shape': np.array([1, 1], dtype=np.int64)
    }))

    return list_of_inputs

generated_inputs["tf.sparse.reshape"] = tf_sparse_reshape_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.reshape' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.reshape'.")

check_valid('tf.sparse.reshape', generated_inputs['tf.sparse.reshape'], lib="tf", suffix=0)
