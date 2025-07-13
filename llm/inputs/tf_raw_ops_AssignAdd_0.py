
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_assign_add_inputs():
    list_of_inputs = []

    # Input 1
    ref = tf.Variable(np.array(1.0, dtype=np.float32))
    value = np.array(2.0, dtype=np.float32)
    use_locking = False
    name = "assign_add_1"
    input_dict = {"ref": ref, "value": value, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    ref = tf.Variable(np.array([1, 2, 3], dtype=np.int32))
    value = np.array([4, 5, 6], dtype=np.int32)
    use_locking = True
    name = "assign_add_2"
    input_dict = {"ref": ref, "value": value, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    ref = tf.Variable(np.array([[1, 2], [3, 4]], dtype=np.int64))
    value = np.array([[5, 6], [7, 8]], dtype=np.int64)
    use_locking = False
    name = "assign_add_3"
    input_dict = {"ref": ref, "value": value, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    ref = tf.Variable(np.array(10, dtype=np.uint8))
    value = np.array(5, dtype=np.uint8)
    use_locking = True
    name = "assign_add_4"
    input_dict = {"ref": ref, "value": value, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    ref = tf.Variable(np.array(1.5, dtype=np.float64))
    value = np.array(-0.5, dtype=np.float64)
    use_locking = False
    name = "assign_add_5"
    input_dict = {"ref": ref, "value": value, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    ref = tf.Variable(np.array([1+1j, 2+2j], dtype=np.complex64))
    value = np.array([3+3j, 4+4j], dtype=np.complex64)
    use_locking = True
    name = "assign_add_6"
    input_dict = {"ref": ref, "value": value, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    ref = tf.Variable(np.array([[1.0+1j, 2.0+2j], [3.0+3j, 4.0+4j]], dtype=np.complex128))
    value = np.array([[5.0+5j, 6.0+6j], [7.0+7j, 8.0+8j]], dtype=np.complex128)
    use_locking = False
    name = "assign_add_7"
    input_dict = {"ref": ref, "value": value, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    ref = tf.Variable(np.array([1, 2, 3], dtype=np.int16))
    value = np.array([4, 5, 6], dtype=np.int16)
    use_locking = True
    name = "assign_add_8"
    input_dict = {"ref": ref, "value": value, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    ref = tf.Variable(np.array([-1, -2, -3], dtype=np.int8))
    value = np.array([4, 5, 6], dtype=np.int8)
    use_locking = False
    name = "assign_add_9"
    input_dict = {"ref": ref, "value": value, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    ref = tf.Variable(np.array([1, 2], dtype=np.uint16))
    value = np.array([3, 4], dtype=np.uint16)
    use_locking = True
    name = "assign_add_10"
    input_dict = {"ref": ref, "value": value, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.AssignAdd"] = tf_raw_ops_assign_add_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.AssignAdd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AssignAdd'.")

check_valid('tf.raw_ops.AssignAdd', generated_inputs['tf.raw_ops.AssignAdd'], lib="tf", suffix=0)
