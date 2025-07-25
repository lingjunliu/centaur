
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

class EagerVariableWrapper:
    """
    A wrapper for tf.Variable to make it compatible with a test harness
    that expects numpy-like attributes (.size) and deepcopy support.
    
    tf.raw_ops.ScatterUpdate requires a Variable-like object (a 'ref' type).
    In Eager execution mode, this op is explicitly disabled and raises an
    error. This wrapper correctly provides the required Variable type,
    which will lead to the expected RuntimeError.
    """
    def __init__(self, initial_value, name=None):
        # The actual tf.Variable that the op will receive.
        self.variable = tf.Variable(initial_value, name=name)
        # A numpy view for harness-required attributes like .size and .shape.
        self._numpy_view = np.array(initial_value)
        self.size = self._numpy_view.size
        self.shape = self._numpy_view.shape
        self.dtype = self._numpy_view.dtype

    def __deepcopy__(self, memo):
        # Create a new instance with a copy of the data.
        if id(self) in memo:
            return memo[id(self)]
        new_copy = EagerVariableWrapper(self._numpy_view.copy())
        memo[id(self)] = new_copy
        return new_copy

def _variable_wrapper_to_tensor(value, dtype=None, name=None, as_ref=False):
    """
    Tensor conversion function to tell TensorFlow how to handle the wrapper.
    It returns the underlying tf.Variable when the wrapper is passed to a TF op.
    """
    return value.variable

# Register the conversion function with a high priority so TF knows what to do
# with EagerVariableWrapper instances.
tf.register_tensor_conversion_function(
    EagerVariableWrapper, _variable_wrapper_to_tensor, priority=100
)

def get_tf_raw_ops_scatter_update_inputs():
    """
    Generates a list of structurally valid inputs for tf.raw_ops.ScatterUpdate.
    NOTE: This raw op is not compatible with Eager execution and is expected to
    raise a RuntimeError when called. The inputs provided here are correct for
    the operation's signature.
    """
    list_of_inputs = []

    # Input 1: Basic 1D update with float32
    input_dict = {
        'ref': EagerVariableWrapper(np.zeros(8, dtype=np.float32)),
        'indices': np.array([4, 3, 1, 7], dtype=np.int32),
        'updates': np.array([9.0, 10.0, 11.0, 12.0], dtype=np.float32),
        'use_locking': True,
        'name': 'basic_float_update'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Updating slices in a 2D tensor
    input_dict = {
        'ref': EagerVariableWrapper(np.zeros((5, 3), dtype=np.float32)),
        'indices': np.array([0, 4, 2], dtype=np.int32),
        'updates': np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        'use_locking': False,
        'name': 'slice_update'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Duplicate indices with int32 and int64 types
    input_dict = {
        'ref': EagerVariableWrapper(np.array([1, 2, 3, 4, 5], dtype=np.int32)),
        'indices': np.array([1, 0, 1, 3], dtype=np.int64),
        'updates': np.array([10, 20, 30, 40], dtype=np.int32),
        'use_locking': True,
        'name': 'duplicate_indices'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: High-rank indices (2D indices)
    input_dict = {
        'ref': EagerVariableWrapper(np.zeros((8, 2), dtype=np.float64)),
        'indices': np.array([[1, 5], [7, 2]], dtype=np.int32),
        'updates': np.arange(8, dtype=np.float64).reshape(2, 2, 2),
        'use_locking': True,
        'name': 'high_rank_indices'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Scalar index and scalar update
    input_dict = {
        'ref': EagerVariableWrapper(np.zeros(5, dtype=np.float32)),
        'indices': np.array(3, dtype=np.int32),
        'updates': np.array(99.0, dtype=np.float32),
        'use_locking': False,
        'name': 'scalar_index_scalar_update'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Scalar index with slice update
    input_dict = {
        'ref': EagerVariableWrapper(np.zeros((5, 4), dtype=np.int32)),
        'indices': np.array(2, dtype=np.int64),
        'updates': np.array([1, 2, 3, 4], dtype=np.int32),
        'use_locking': True,
        'name': 'scalar_index_slice_update'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty indices and updates
    input_dict = {
        'ref': EagerVariableWrapper(np.array([1.0, 2.0, 3.0], dtype=np.float32)),
        'indices': np.array([], dtype=np.int32),
        'updates': np.array([], dtype=np.float32),
        'use_locking': True,
        'name': 'empty_update'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Higher dimensional ref (3D)
    input_dict = {
        'ref': EagerVariableWrapper(np.ones((4, 3, 2), dtype=np.float32)),
        'indices': np.array([0, 3], dtype=np.int32),
        'updates': np.zeros((2, 3, 2), dtype=np.float32),
        'use_locking': False,
        'name': '3d_ref_update'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: All indices used in a different order
    input_dict = {
        'ref': EagerVariableWrapper(np.zeros(4, dtype=np.int64)),
        'indices': np.array([3, 1, 0, 2], dtype=np.int32),
        'updates': np.array([10, 20, 30, 40], dtype=np.int64),
        'use_locking': True,
        'name': 'full_update_shuffled'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Scalar update broadcast to slices
    input_dict = {
        'ref': EagerVariableWrapper(np.zeros((5, 3), dtype=np.float32)),
        'indices': np.array([1, 4], dtype=np.int32),
        'updates': np.array(7.7, dtype=np.float32),
        'use_locking': True,
        'name': 'scalar_broadcast_update'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.ScatterUpdate"] = get_tf_raw_ops_scatter_update_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ScatterUpdate' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScatterUpdate'.")

check_valid('tf.raw_ops.ScatterUpdate', generated_inputs['tf.raw_ops.ScatterUpdate'], lib="tf", suffix=0)
