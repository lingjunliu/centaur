
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_assign_inputs():
    list_of_inputs = []

    # Input 1: Basic assignment
    ref_np = np.array(1, dtype=np.int32)
    value_np = np.array(2, dtype=np.int32)
    ref = tf.Variable(ref_np)
    value = tf.constant(value_np)
    input_dict = {"ref": ref, "value": value, "validate_shape": True, "use_locking": True, "name": "assign_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different shape, validate_shape=False
    ref_np = np.array([1, 2], dtype=np.int32)
    value_np = np.array([3, 4, 5], dtype=np.int32)
    ref = tf.Variable(ref_np)
    value = tf.constant(value_np)
    input_dict = {"ref": ref, "value": value, "validate_shape": False, "use_locking": False, "name": "assign_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multi-dimensional array
    ref_np = np.array([[1, 2], [3, 4]], dtype=np.float32)
    value_np = np.array([[5, 6], [7, 8]], dtype=np.float32)
    ref = tf.Variable(ref_np)
    value = tf.constant(value_np)
    input_dict = {"ref": ref, "value": value, "validate_shape": True, "use_locking": True, "name": "assign_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Boolean array
    ref_np = np.array([True, False], dtype=np.bool_)
    value_np = np.array([False, True], dtype=np.bool_)
    ref = tf.Variable(ref_np)
    value = tf.constant(value_np)
    input_dict = {"ref": ref, "value": value, "validate_shape": True, "use_locking": False, "name": "assign_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: String array
    ref_np = np.array(["a", "b"], dtype=np.string_)
    value_np = np.array(["c", "d"], dtype=np.string_)
    ref = tf.Variable(ref_np)
    value = tf.constant(value_np)
    input_dict = {"ref": ref, "value": value, "validate_shape": True, "use_locking": True, "name": "assign_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative values
    ref_np = np.array([-1, -2], dtype=np.int32)
    value_np = np.array([-3, -4], dtype=np.int32)
    ref = tf.Variable(ref_np)
    value = tf.constant(value_np)
    input_dict = {"ref": ref, "value": value, "validate_shape": True, "use_locking": False, "name": "assign_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different dtype (float64)
    ref_np = np.array(1.0, dtype=np.float64)
    value_np = np.array(2.0, dtype=np.float64)
    ref = tf.Variable(ref_np)
    value = tf.constant(value_np)
    input_dict = {"ref": ref, "value": value, "validate_shape": True, "use_locking": False, "name": "assign_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: More complex shape
    ref_np = np.zeros((2, 3, 4), dtype=np.int32)
    value_np = np.ones((2, 3, 4), dtype=np.int32)
    ref = tf.Variable(ref_np)
    value = tf.constant(value_np)
    input_dict = {"ref": ref, "value": value, "validate_shape": True, "use_locking": True, "name": "assign_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: validate_shape = False, different shape (again)
    ref_np = np.array([1, 2, 3], dtype=np.int32)
    value_np = np.array([4, 5], dtype=np.int32)
    ref = tf.Variable(ref_np)
    value = tf.constant(value_np)
    input_dict = {"ref": ref, "value": value, "validate_shape": False, "use_locking": False, "name": "assign_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Scalar assignment
    ref_np = np.array(1.5, dtype=np.float32)
    value_np = np.array(3.7, dtype=np.float32)
    ref = tf.Variable(ref_np)
    value = tf.constant(value_np)
    input_dict = {"ref": ref, "value": value, "validate_shape": True, "use_locking": True, "name": "assign_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Assigning a scalar value to a tensor
    ref_np = np.array([1, 2, 3], dtype=np.int32)
    value_np = np.array(5, dtype=np.int32)
    ref = tf.Variable(ref_np)
    value = tf.constant(value_np)
    input_dict = {"ref": ref, "value": value, "validate_shape": False, "use_locking": True, "name": "assign_11"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Assigning a tensor to a tensor of same shape
    ref_np = np.array([[1, 2], [3, 4]], dtype=np.float32)
    value_np = np.array([[5, 6], [7, 8]], dtype=np.float32)
    ref = tf.Variable(ref_np)
    value = tf.constant(value_np)
    input_dict = {"ref": ref, "value": value, "validate_shape": True, "use_locking": True, "name": "assign_12"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Assign"] = tf_raw_ops_assign_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Assign' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Assign'.")

check_valid('tf.raw_ops.Assign', generated_inputs['tf.raw_ops.Assign'], lib="tf", suffix=0)
