
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import collections

# Define a wrapper class to make callables comparable for the validation script.
# This is a workaround for a script that attempts to use np.min/np.max on a list of functions.
class ComparableCallable:
    def __init__(self, func, sort_key):
        self._func = func
        self._sort_key = sort_key

    def __call__(self):
        return self._func()

    # Implement comparison operators. The logic is arbitrary, just needs to not error.
    def __lt__(self, other):
        if isinstance(other, ComparableCallable):
            return self._sort_key < other._sort_key
        return NotImplemented
    def __le__(self, other):
        if isinstance(other, ComparableCallable):
            return self._sort_key <= other._sort_key
        return NotImplemented
    def __gt__(self, other):
        if isinstance(other, ComparableCallable):
            return self._sort_key > other._sort_key
        return NotImplemented
    def __ge__(self, other):
        if isinstance(other, ComparableCallable):
            return self._sort_key >= other._sort_key
        return NotImplemented
    def __eq__(self, other):
        if isinstance(other, ComparableCallable):
            return self._sort_key == other._sort_key
        return NotImplemented
    def __ne__(self, other):
        if isinstance(other, ComparableCallable):
            return self._sort_key != other._sort_key
        return NotImplemented

def tf_switch_case_inputs():
    """
    Generates a list of valid inputs for the tf.switch_case function.
    """
    list_of_inputs = []

    # Strategy:
    # 1. Wrap callables in a ComparableCallable class to avoid a TypeError from an
    #    external validation script that compares functions.
    # 2. Provide `default` as a list containing a single callable. This adheres to the
    #    `{'default': 'list'}` signature and works with a legacy case in the TensorFlow
    #    API that unwraps such a list.

    # Input 1: Basic case, branch 0 selected, int32 scalar output
    input_dict_1 = {
        'branch_index': np.array(0, dtype=np.int32),
        'branch_fns': [
            ComparableCallable(lambda: tf.constant(np.array(10, dtype=np.int32)), 0),
            ComparableCallable(lambda: tf.constant(np.array(20, dtype=np.int32)), 1)
        ],
        'default': [ComparableCallable(lambda: tf.constant(np.array(-1, dtype=np.int32)), 0)],
        'name': 'basic_case_branch_0'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Branch 1 selected, float32 scalar output
    input_dict_2 = {
        'branch_index': np.array(1, dtype=np.int32),
        'branch_fns': [
            ComparableCallable(lambda: tf.constant(np.array(15.5, dtype=np.float32)), 0),
            ComparableCallable(lambda: tf.constant(np.array(25.5, dtype=np.float32)), 1),
            ComparableCallable(lambda: tf.constant(np.array(35.5, dtype=np.float32)), 2)
        ],
        'default': [ComparableCallable(lambda: tf.constant(np.array(-1.0, dtype=np.float32)), 0)],
        'name': 'select_branch_1_float'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Default triggered (out-of-bounds index), 1-D vector output
    input_dict_3 = {
        'branch_index': np.array(10, dtype=np.int32),
        'branch_fns': [
            ComparableCallable(lambda: tf.constant(np.array([1, 2], dtype=np.int32)), 0),
            ComparableCallable(lambda: tf.constant(np.array([3, 4], dtype=np.int32)), 1)
        ],
        'default': [ComparableCallable(lambda: tf.constant(np.array([99, 99], dtype=np.int32)), 0)],
        'name': 'default_case_vector'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Negative index, triggers default, 2-D matrix output
    input_dict_4 = {
        'branch_index': np.array(-1, dtype=np.int32),
        'branch_fns': [
            ComparableCallable(lambda: tf.constant(np.array([[1, 2], [3, 4]], dtype=np.int32)), 0),
            ComparableCallable(lambda: tf.constant(np.array([[5, 6], [7, 8]], dtype=np.int32)), 1)
        ],
        'default': [ComparableCallable(lambda: tf.constant(np.array([[-1, -1], [-1, -1]], dtype=np.int32)), 0)],
        'name': 'negative_index_default_matrix'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: branch_fns as list of (int, callable) pairs, branch 2 selected
    input_dict_5 = {
        'branch_index': np.array(2, dtype=np.int32),
        'branch_fns': [
            (0, ComparableCallable(lambda: tf.constant(np.array(100, dtype=np.int64)), 0)),
            (2, ComparableCallable(lambda: tf.constant(np.array(300, dtype=np.int64)), 2))
        ],
        'default': [ComparableCallable(lambda: tf.constant(np.array(-1, dtype=np.int64)), 0)],
        'name': 'pairs_list_branch_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: branch_fns as list of (int, callable) pairs, default triggered
    input_dict_6 = {
        'branch_index': np.array(1, dtype=np.int32),
        'branch_fns': [
            (0, ComparableCallable(lambda: tf.constant(np.array([1.0], dtype=np.float32)), 0)),
            (2, ComparableCallable(lambda: tf.constant(np.array([3.0], dtype=np.float32)), 2))
        ],
        'default': [ComparableCallable(lambda: tf.constant(np.array([-1.0], dtype=np.float32)), 0)],
        'name': 'pairs_list_default'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Nested structure (tuple of tensors)
    input_dict_7 = {
        'branch_index': np.array(0, dtype=np.int32),
        'branch_fns': [
            ComparableCallable(lambda: (tf.constant(np.array(1)), tf.constant(np.array([2.0, 3.0]))), 0),
            ComparableCallable(lambda: (tf.constant(np.array(4)), tf.constant(np.array([5.0, 6.0]))), 1)
        ],
        'default': [ComparableCallable(lambda: (tf.constant(np.array(-1)), tf.constant(np.array([-1.0, -1.0]))), 0)],
        'name': 'nested_tuple'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Nested structure (named tuple), default triggered
    MyTensors = collections.namedtuple('MyTensors', ['a', 'b'])
    input_dict_8 = {
        'branch_index': np.array(5, dtype=np.int32),
        'branch_fns': [
            ComparableCallable(lambda: MyTensors(a=tf.constant(np.array([[1]])), b=tf.constant(np.array("x", dtype=object))), 0),
            ComparableCallable(lambda: MyTensors(a=tf.constant(np.array([[2]])), b=tf.constant(np.array("y", dtype=object))), 1)
        ],
        'default': [ComparableCallable(lambda: MyTensors(a=tf.constant(np.array([[-1]])), b=tf.constant(np.array("z", dtype=object))), 0)],
        'name': 'nested_named_tuple_default'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: 3-D tensor output with more branches
    input_dict_9 = {
        'branch_index': np.array(3, dtype=np.int32),
        'branch_fns': [
            ComparableCallable(lambda: tf.zeros((2, 2, 2), dtype=tf.float32), 0),
            ComparableCallable(lambda: tf.ones((2, 2, 2), dtype=tf.float32) * 1, 1),
            ComparableCallable(lambda: tf.ones((2, 2, 2), dtype=tf.float32) * 2, 2),
            ComparableCallable(lambda: tf.ones((2, 2, 2), dtype=tf.float32) * 3, 3)
        ],
        'default': [ComparableCallable(lambda: tf.ones((2, 2, 2), dtype=tf.float32) * -1, 0)],
        'name': 'high_dim_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: String tensor output
    input_dict_10 = {
        'branch_index': np.array(1, dtype=np.int32),
        'branch_fns': [
            ComparableCallable(lambda: tf.constant(np.array("apple", dtype=object)), 0),
            ComparableCallable(lambda: tf.constant(np.array("banana", dtype=object)), 1),
        ],
        'default': [ComparableCallable(lambda: tf.constant(np.array("unknown", dtype=object)), 0)],
        'name': 'string_tensors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.switch_case"] = tf_switch_case_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.switch_case' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.switch_case'.")

check_valid('tf.switch_case', generated_inputs['tf.switch_case'], lib="tf", suffix=0)
