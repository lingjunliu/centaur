
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_assign_inputs():
    list_of_inputs = []

    # Input 1: Basic assignment with shape validation
    ref = tf.Variable(np.array([1, 2, 3], dtype=np.int32)).value()
    value = np.array([4, 5, 6], dtype=np.int32)
    input_dict = {"ref": ref, "value": value, "validate_shape": True, "use_locking": True, "name": "assign_op_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Assignment without shape validation
    ref = tf.Variable(np.array([1, 2, 3], dtype=np.int32)).value()
    value = np.array([4, 5, 6, 7], dtype=np.int32)
    input_dict = {"ref": ref, "value": value, "validate_shape": False, "use_locking": False, "name": "assign_op_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Assignment with float32 data type
    ref = tf.Variable(np.array([1.0, 2.0, 3.0], dtype=np.float32)).value()
    value = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    input_dict = {"ref": ref, "value": value, "validate_shape": True, "use_locking": True, "name": "assign_op_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Assignment with multi-dimensional array
    ref = tf.Variable(np.array([[1, 2], [3, 4]], dtype=np.int32)).value()
    value = np.array([[5, 6], [7, 8]], dtype=np.int32)
    input_dict = {"ref": ref, "value": value, "validate_shape": True, "use_locking": False, "name": "assign_op_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Assignment with bool data type
    ref = tf.Variable(np.array([True, False, True], dtype=np.bool_)).value()
    value = np.array([False, True, False], dtype=np.bool_)
    input_dict = {"ref": ref, "value": value, "validate_shape": True, "use_locking": False, "name": "assign_op_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6: Assignment with int64 data type
    ref = tf.Variable(np.array([1, 2, 3], dtype=np.int64)).value()
    value = np.array([4, 5, 6], dtype=np.int64)
    input_dict = {"ref": ref, "value": value, "validate_shape": True, "use_locking": True, "name": "assign_op_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Assignment with shape (1,1)
    ref = tf.Variable(np.array([[1]], dtype=np.int32)).value()
    value = np.array([[2]], dtype=np.int32)
    input_dict = {"ref": ref, "value": value, "validate_shape": True, "use_locking": True, "name": "assign_op_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: different shape with no validation
    ref = tf.Variable(np.array([1, 2], dtype=np.int32)).value()
    value = np.array([3, 4, 5], dtype=np.int32)
    input_dict = {"ref": ref, "value": value, "validate_shape": False, "use_locking": True, "name": "assign_op_11"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D tensor
    ref = tf.Variable(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)).value()
    value = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.int32)
    input_dict = {"ref": ref, "value": value, "validate_shape": True, "use_locking": False, "name": "assign_op_12"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: complex64
    ref = tf.Variable(np.array([1+1j, 2+2j], dtype=np.complex64)).value()
    value = np.array([3+3j, 4+4j], dtype=np.complex64)
    input_dict = {"ref": ref, "value": value, "validate_shape": True, "use_locking": True, "name": "assign_op_13"}
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
