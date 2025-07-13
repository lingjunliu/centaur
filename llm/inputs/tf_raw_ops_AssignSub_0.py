
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_assign_sub_inputs():
    list_of_inputs = []

    # Input 1: float32, no locking
    ref = tf.Variable(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    value = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    use_locking = False
    name = "assign_sub_float32_no_lock"
    input_dict = {"ref": ref, "value": value, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int32, with locking
    ref = tf.Variable(np.array([4, 5, 6], dtype=np.int32))
    value = np.array([1, 2, 3], dtype=np.int32)
    use_locking = True
    name = "assign_sub_int32_lock"
    input_dict = {"ref": ref, "value": value, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64, no locking, different shape
    ref = tf.Variable(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
    value = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    use_locking = False
    name = "assign_sub_float64_no_lock_diff_shape"
    input_dict = {"ref": ref, "value": value, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int64, with locking, negative value
    ref = tf.Variable(np.array([7, 8, 9], dtype=np.int64))
    value = np.array([-1, -2, -3], dtype=np.int64)
    use_locking = True
    name = "assign_sub_int64_lock_negative"
    input_dict = {"ref": ref, "value": value, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: uint8, no locking
    ref = tf.Variable(np.array([10, 11, 12], dtype=np.uint8))
    value = np.array([2, 3, 4], dtype=np.uint8)
    use_locking = False
    name = "assign_sub_uint8_no_lock"
    input_dict = {"ref": ref, "value": value, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex64, with locking
    ref = tf.Variable(np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64))
    value = np.array([0.5+0.5j, 1+1j, 1.5+1.5j], dtype=np.complex64)
    use_locking = True
    name = "assign_sub_complex64_lock"
    input_dict = {"ref": ref, "value": value, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7: uint32, with locking
    ref = tf.Variable(np.array([100, 101, 102], dtype=np.uint32))
    value = np.array([10, 11, 12], dtype=np.uint32)
    use_locking = True
    name = "assign_sub_uint32_lock"
    input_dict = {"ref": ref, "value": value, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: uint64, no locking
    ref = tf.Variable(np.array([2**10, 2**10+1, 2**10+2], dtype=np.uint64))
    value = np.array([100, 101, 102], dtype=np.uint64)
    use_locking = False
    name = "assign_sub_uint64_no_lock"
    input_dict = {"ref": ref, "value": value, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: qint32, with locking
    ref = tf.Variable(np.array([10, 11, 12], dtype=np.int32))
    value = np.array([2, 3, 4], dtype=np.int32)
    use_locking = True
    name = "assign_sub_qint32_lock"
    input_dict = {"ref": ref, "value": value, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: half, no locking
    ref = tf.Variable(np.array([1.0, 2.0, 3.0], dtype=np.float16))
    value = np.array([0.5, 1.0, 1.5], dtype=np.float16)
    use_locking = False
    name = "assign_sub_half_no_lock"
    input_dict = {"ref": ref, "value": value, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.AssignSub"] = tf_raw_ops_assign_sub_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.AssignSub' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AssignSub'.")

check_valid('tf.raw_ops.AssignSub', generated_inputs['tf.raw_ops.AssignSub'], lib="tf", suffix=0)
