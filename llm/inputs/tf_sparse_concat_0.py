
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

class TensorList(np.ndarray):
    def __new__(cls, tensors):
        obj = np.arange(len(tensors), dtype=np.int32).view(cls)
        obj._tensors = tensors
        return obj

    def __iter__(self):
        return iter(self._tensors)

    def __len__(self):
        return len(self._tensors)

    def __getitem__(self, idx):
        return self._tensors[idx]
    
    @property
    def dtype(self):
        if self._tensors:
            for tensor in self._tensors:
                if len(tensor.values) > 0:
                    return tensor.values.dtype
        return tf.float32

def tf_sparse_concat_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D concatenation along axis=1
    sp_input1_1 = tf.SparseTensor(
        indices=np.array([[0, 2], [1, 0], [1, 1]], dtype=np.int64),
        values=np.array([1, 2, 3], dtype=np.int32),
        dense_shape=np.array([2, 3], dtype=np.int64)
    )
    sp_input1_2 = tf.SparseTensor(
        indices=np.array([[0, 1], [0, 2]], dtype=np.int64),
        values=np.array([4, 5], dtype=np.int32),
        dense_shape=np.array([2, 4], dtype=np.int64)
    )
    list_of_inputs.append({
        'axis': 1,
        'sp_inputs': TensorList([sp_input1_1, sp_input1_2]),
        'expand_nonconcat_dims': False,
        'name': 'basic_concat_axis1'
    })

    # Input 2: Expand non-concat dims
    sp_input2_1 = tf.SparseTensor(
        indices=np.array([[0, 2], [1, 0], [2, 1]], dtype=np.int64),
        values=np.array([10, 20, 30], dtype=np.int32),
        dense_shape=np.array([3, 3], dtype=np.int64)
    )
    sp_input2_2 = tf.SparseTensor(
        indices=np.array([[0, 1], [0, 2]], dtype=np.int64),
        values=np.array([40, 50], dtype=np.int32),
        dense_shape=np.array([2, 4], dtype=np.int64)
    )
    list_of_inputs.append({
        'axis': 1,
        'sp_inputs': TensorList([sp_input2_1, sp_input2_2]),
        'expand_nonconcat_dims': True,
        'name': 'expand_dims_concat'
    })

    # Input 3: Concatenation along axis=0
    sp_input3_1 = tf.SparseTensor(
        indices=np.array([[0, 1], [1, 3]], dtype=np.int64),
        values=np.array([10, 20], dtype=np.int32),
        dense_shape=np.array([2, 5], dtype=np.int64)
    )
    sp_input3_2 = tf.SparseTensor(
        indices=np.array([[0, 0], [2, 4]], dtype=np.int64),
        values=np.array([30, 40], dtype=np.int32),
        dense_shape=np.array([3, 5], dtype=np.int64)
    )
    list_of_inputs.append({
        'axis': 0,
        'sp_inputs': TensorList([sp_input3_1, sp_input3_2]),
        'expand_nonconcat_dims': False,
        'name': 'concat_axis0_int'
    })

    # Input 4: Higher rank (3D) concatenation with float values
    sp_input4_1 = tf.SparseTensor(
        indices=np.array([[0, 1, 1], [1, 2, 0]], dtype=np.int64),
        values=np.array([1.1, 2.2], dtype=np.float32),
        dense_shape=np.array([2, 3, 4], dtype=np.int64)
    )
    sp_input4_2 = tf.SparseTensor(
        indices=np.array([[0, 0, 3], [1, 1, 1]], dtype=np.int64),
        values=np.array([3.3, 4.4], dtype=np.float32),
        dense_shape=np.array([2, 3, 5], dtype=np.int64)
    )
    list_of_inputs.append({
        'axis': 2,
        'sp_inputs': TensorList([sp_input4_1, sp_input4_2]),
        'expand_nonconcat_dims': False,
        'name': '3d_concat_float'
    })

    # Input 5: Negative axis
    sp_input5_1 = tf.SparseTensor(
        indices=np.array([[0, 1, 1]], dtype=np.int64),
        values=np.array([100], dtype=np.int64),
        dense_shape=np.array([2, 2, 2], dtype=np.int64)
    )
    sp_input5_2 = tf.SparseTensor(
        indices=np.array([[1, 0, 0]], dtype=np.int64),
        values=np.array([200], dtype=np.int64),
        dense_shape=np.array([2, 2, 3], dtype=np.int64)
    )
    list_of_inputs.append({
        'axis': -1,
        'sp_inputs': TensorList([sp_input5_1, sp_input5_2]),
        'expand_nonconcat_dims': False,
        'name': 'negative_axis'
    })

    # Input 6: More than two inputs
    sp_input6_1 = tf.SparseTensor(
        indices=np.array([[0, 4]], dtype=np.int64),
        values=np.array([100], dtype=np.int32),
        dense_shape=np.array([1, 5], dtype=np.int64)
    )
    sp_input6_2 = tf.SparseTensor(
        indices=np.array([[0, 0], [1, 2]], dtype=np.int64),
        values=np.array([200, 300], dtype=np.int32),
        dense_shape=np.array([2, 5], dtype=np.int64)
    )
    sp_input6_3 = tf.SparseTensor(
        indices=np.array([[0, 1]], dtype=np.int64),
        values=np.array([400], dtype=np.int32),
        dense_shape=np.array([1, 5], dtype=np.int64)
    )
    list_of_inputs.append({
        'axis': 0,
        'sp_inputs': TensorList([sp_input6_1, sp_input6_2, sp_input6_3]),
        'expand_nonconcat_dims': False,
        'name': 'multiple_inputs'
    })

    # Input 7: One empty input
    sp_input7_1 = tf.SparseTensor(
        indices=np.array([[0, 0]], dtype=np.int64),
        values=np.array([1.0], dtype=np.float64),
        dense_shape=np.array([2, 1], dtype=np.int64)
    )
    sp_input7_2 = tf.SparseTensor(
        indices=np.empty((0, 2), dtype=np.int64),
        values=np.array([], dtype=np.float64),
        dense_shape=np.array([2, 3], dtype=np.int64)
    )
    list_of_inputs.append({
        'axis': 1,
        'sp_inputs': TensorList([sp_input7_1, sp_input7_2]),
        'expand_nonconcat_dims': False,
        'name': 'one_empty_input'
    })

    # Input 8: All empty inputs
    sp_input8_1 = tf.SparseTensor(
        indices=np.empty((0, 2), dtype=np.int64),
        values=np.array([], dtype=np.float32),
        dense_shape=np.array([2, 5], dtype=np.int64)
    )
    sp_input8_2 = tf.SparseTensor(
        indices=np.empty((0, 2), dtype=np.int64),
        values=np.array([], dtype=np.float32),
        dense_shape=np.array([3, 5], dtype=np.int64)
    )
    list_of_inputs.append({
        'axis': 0,
        'sp_inputs': TensorList([sp_input8_1, sp_input8_2]),
        'expand_nonconcat_dims': False,
        'name': 'all_empty_inputs'
    })

    # Input 9: 1D tensors
    sp_input9_1 = tf.SparseTensor(
        indices=np.array([[2], [4]], dtype=np.int64),
        values=np.array([1, 2], dtype=np.int32),
        dense_shape=np.array([5], dtype=np.int64)
    )
    sp_input9_2 = tf.SparseTensor(
        indices=np.array([[0], [3]], dtype=np.int64),
        values=np.array([3, 4], dtype=np.int32),
        dense_shape=np.array([6], dtype=np.int64)
    )
    list_of_inputs.append({
        'axis': 0,
        'sp_inputs': TensorList([sp_input9_1, sp_input9_2]),
        'expand_nonconcat_dims': False,
        'name': '1d_concat'
    })

    # Input 10: Expand non-concat dims with multiple different non-concat dimensions
    sp_input10_1 = tf.SparseTensor(
        indices=np.array([[0, 1, 1], [1, 2, 0]], dtype=np.int64),
        values=np.array([1.1, 2.2], dtype=np.float32),
        dense_shape=np.array([2, 3, 4], dtype=np.int64)
    )
    sp_input10_2 = tf.SparseTensor(
        indices=np.array([[0, 0, 3], [2, 1, 1]], dtype=np.int64),
        values=np.array([3.3, 4.4], dtype=np.float32),
        dense_shape=np.array([3, 2, 5], dtype=np.int64)
    )
    list_of_inputs.append({
        'axis': 2,
        'sp_inputs': TensorList([sp_input10_1, sp_input10_2]),
        'expand_nonconcat_dims': True,
        'name': 'expand_multiple_dims'
    })

    return list_of_inputs

generated_inputs["tf.sparse.concat"] = tf_sparse_concat_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.concat' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.concat'.")

check_valid('tf.sparse.concat', generated_inputs['tf.sparse.concat'], lib="tf", suffix=0)
