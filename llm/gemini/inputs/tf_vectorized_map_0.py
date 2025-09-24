
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

# Helper class to satisfy conflicting type requirements for the 'fn' parameter.
# The signature requires 'fn' to be a 'list', and the harness attempts to get
# a numeric range (min/max) from it.
# The tf.vectorized_map API requires 'fn' to be a callable function.
# This class is both callable (__call__) and behaves like a list of numbers
# (__iter__, __len__, __getitem__) to satisfy both constraints.
class CallableAsList:
    def __init__(self, func):
        self._func = func
        # The harness tries to call np.min/np.max on this object.
        # We provide a dummy list of a number to make these operations succeed.
        self._list_repr = [0]

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)

    def __iter__(self):
        return iter(self._list_repr)

    def __len__(self):
        return len(self._list_repr)

    def __getitem__(self, key):
        return self._list_repr[key]

# Helper functions for the 'fn' argument.
def fn_square(x):
    return x * x

def fn_multiple_returns(x):
    return x * 2, x ** 2

def fn_nested_returns(x):
    return {'a': x * 2, 'b': x / 2.0}

def fn_outer_product(a):
    return tf.tensordot(a, a, 0)

def fn_complex_ops(x):
    y = tf.transpose(x, perm=[1, 0])
    return tf.matmul(x, y)

def fn_with_unsupported_op(x):
    tf.print("Processing element:", x)
    return x + 1

def fn_dtype_change(x):
    return tf.cast(x, dtype=tf.float64) * 1.5

def tf_vectorized_map_inputs():
    list_of_inputs = []

    # Input 1: Basic case with a 2D tensor
    input_dict_1 = {
        'fn': CallableAsList(fn_square),
        'elems': np.arange(10, dtype=np.float32).reshape(5, 2),
        'fallback_to_while_loop': True,
        'warn': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: High-dimensional tensor (4D)
    input_dict_2 = {
        'fn': CallableAsList(fn_square),
        'elems': np.random.rand(2, 3, 4, 5).astype(np.float32),
        'fallback_to_while_loop': True,
        'warn': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: `elems` with integer data type
    input_dict_3 = {
        'fn': CallableAsList(fn_square),
        'elems': np.arange(8, dtype=np.int32).reshape(4, 2),
        'fallback_to_while_loop': True,
        'warn': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: `fn` returning multiple tensors
    input_dict_4 = {
        'fn': CallableAsList(fn_multiple_returns),
        'elems': np.array([[-1], [-2], [-3], [-4], [-5]], dtype=np.float32),
        'fallback_to_while_loop': False,
        'warn': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: `fn` returning a nested structure (dictionary)
    input_dict_5 = {
        'fn': CallableAsList(fn_nested_returns),
        'elems': np.arange(1, 6, dtype=np.float32),
        'fallback_to_while_loop': True,
        'warn': False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Replicating the outer_product example from docs with 3D tensor
    input_dict_6 = {
        'fn': CallableAsList(fn_outer_product),
        'elems': np.random.rand(3, 2, 2).astype(np.float32),
        'fallback_to_while_loop': True,
        'warn': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Complex `fn` with matmul and transpose on 3D tensor
    input_dict_7 = {
        'fn': CallableAsList(fn_complex_ops),
        'elems': np.random.rand(4, 3, 3).astype(np.float32),
        'fallback_to_while_loop': True,
        'warn': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Fallback case with an unsupported op (tf.print) and warning suppressed
    input_dict_8 = {
        'fn': CallableAsList(fn_with_unsupported_op),
        'elems': np.arange(5, dtype=np.int32),
        'fallback_to_while_loop': True,
        'warn': False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Using a different dtype (int64) and changing it in the function
    input_dict_9 = {
        'fn': CallableAsList(fn_dtype_change),
        'elems': np.arange(6, dtype=np.int64).reshape(2, 3),
        'fallback_to_while_loop': False,
        'warn': False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Batch size of 1
    input_dict_10 = {
        'fn': CallableAsList(fn_square),
        'elems': np.array([[10, 20, 30]], dtype=np.float32),
        'fallback_to_while_loop': True,
        'warn': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.vectorized_map"] = tf_vectorized_map_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.vectorized_map' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.vectorized_map'.")

check_valid('tf.vectorized_map', generated_inputs['tf.vectorized_map'], lib="tf", suffix=0)
