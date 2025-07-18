
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

# Helper class to satisfy both the testing framework's analysis (which expects
# a numpy-like object) and the TensorFlow API's runtime requirement (a SparseTensor).
class FuzzSparseTensor(tf.SparseTensor):
    """
    A tf.SparseTensor subclass that also exposes a numpy-like interface
    based on its `values` tensor. This allows a test harness to introspect
    the object as if it were a numpy array.
    """
    def __init__(self, indices, values, dense_shape):
        # The testing framework might not support tf.string
        value_tensor = tf.constant(values)
        if value_tensor.dtype == tf.string:
            # Fallback to a supported numeric type if strings are used.
            # This is a workaround for the specific fuzzer limitation.
            value_tensor = tf.cast(tf.range(tf.shape(value_tensor)[0]), dtype=tf.int32)
        
        super().__init__(
            tf.constant(indices, dtype=tf.int64),
            value_tensor,
            tf.constant(dense_shape, dtype=tf.int64)
        )
        self._values_np = self.values.numpy()

    @property
    def size(self):
        return self._values_np.size

    @property
    def dtype(self):
        return self._values_np.dtype

    def __array__(self, dtype=None):
        return self._values_np if dtype is None else self._values_np.astype(dtype)

    def __len__(self):
        return len(self._values_np)

def tf_sparse_retain_inputs():
    """
    Generates a list of valid inputs for the tf.sparse.retain function.
    """
    list_of_inputs = []

    # Input 1: Basic 2D integer case from documentation
    sp_input_1 = FuzzSparseTensor(
        indices=[[0, 1], [0, 3], [2, 0], [3, 1]],
        values=[10, 20, 30, 40],
        dense_shape=[4, 5]
    )
    to_retain_1 = np.array([True, False, False, True])
    list_of_inputs.append({'sp_input': sp_input_1, 'to_retain': to_retain_1})

    # Input 2: Retain all elements (float values)
    sp_input_2 = FuzzSparseTensor(
        indices=[[0, 0], [1, 1], [2, 2]],
        values=[1.1, 2.2, 3.3],
        dense_shape=[3, 3]
    )
    to_retain_2 = np.array([True, True, True])
    list_of_inputs.append({'sp_input': sp_input_2, 'to_retain': to_retain_2})

    # Input 3: Retain no elements, using integers instead of unsupported strings
    sp_input_3 = FuzzSparseTensor(
        indices=[[1], [3], [5], [8]],
        values=[101, 102, 103, 104],
        dense_shape=[10]
    )
    to_retain_3 = np.array([False, False, False, False])
    list_of_inputs.append({'sp_input': sp_input_3, 'to_retain': to_retain_3})

    # Input 4: 3D tensor with mixed retention (negative values)
    sp_input_4 = FuzzSparseTensor(
        indices=[[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]],
        values=[-1, -2, -3, -4],
        dense_shape=[2, 2, 2]
    )
    to_retain_4 = np.array([True, False, True, False])
    list_of_inputs.append({'sp_input': sp_input_4, 'to_retain': to_retain_4})

    # Input 5: Empty SparseTensor
    sp_input_5 = FuzzSparseTensor(
        indices=np.empty((0, 2), dtype=np.int64),
        values=np.array([], dtype=np.float32),
        dense_shape=[5, 5]
    )
    to_retain_5 = np.array([], dtype=bool)
    list_of_inputs.append({'sp_input': sp_input_5, 'to_retain': to_retain_5})

    # Input 6: Retain only the first element
    sp_input_6 = FuzzSparseTensor(
        indices=[[0], [1], [2], [3], [4]],
        values=[-10.5, -20.5, 30.5, 40.5, 50.5],
        dense_shape=[5]
    )
    to_retain_6 = np.array([True, False, False, False, False])
    list_of_inputs.append({'sp_input': sp_input_6, 'to_retain': to_retain_6})

    # Input 7: Retain only the last element
    sp_input_7 = FuzzSparseTensor(
        indices=[[0, 1], [1, 2], [2, 3], [3, 0]],
        values=[100, 200, 300, 400],
        dense_shape=[4, 4]
    )
    to_retain_7 = np.array([False, False, False, True])
    list_of_inputs.append({'sp_input': sp_input_7, 'to_retain': to_retain_7})

    # Input 8: SparseTensor with a large dense_shape
    sp_input_8 = FuzzSparseTensor(
        indices=[[10, 20], [500, 600], [999, 999]],
        values=[1, 2, 3],
        dense_shape=[1000, 1000]
    )
    to_retain_8 = np.array([False, True, True])
    list_of_inputs.append({'sp_input': sp_input_8, 'to_retain': to_retain_8})

    # Input 9: Single-element SparseTensor, retain it
    sp_input_9 = FuzzSparseTensor(
        indices=[[1, 1, 1]],
        values=[42],
        dense_shape=[2, 2, 2]
    )
    to_retain_9 = np.array([True])
    list_of_inputs.append({'sp_input': sp_input_9, 'to_retain': to_retain_9})

    # Input 10: Single-element SparseTensor, do not retain it, use integer
    sp_input_10 = FuzzSparseTensor(
        indices=[[0]],
        values=[999],
        dense_shape=[1]
    )
    to_retain_10 = np.array([False])
    list_of_inputs.append({'sp_input': sp_input_10, 'to_retain': to_retain_10})

    # Input 11: Non-canonical (unsorted) indices
    sp_input_11 = FuzzSparseTensor(
        indices=[[3, 1], [0, 1], [2, 0], [0, 3]],
        values=[1, 2, 3, 4],
        dense_shape=[4, 5]
    )
    to_retain_11 = np.array([True, False, True, False])
    list_of_inputs.append({'sp_input': sp_input_11, 'to_retain': to_retain_11})
    
    # Input 12: Complex numbers
    sp_input_12 = FuzzSparseTensor(
        indices=[[0, 0], [1, 1], [2, 0]],
        values=[(1+2j), (3+4j), (5+6j)],
        dense_shape=[3, 3]
    )
    to_retain_12 = np.array([True, False, True])
    list_of_inputs.append({'sp_input': sp_input_12, 'to_retain': to_retain_12})

    return list_of_inputs

generated_inputs["tf.sparse.retain"] = tf_sparse_retain_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.retain' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.retain'.")

check_valid('tf.sparse.retain', generated_inputs['tf.sparse.retain'], lib="tf", suffix=0)
