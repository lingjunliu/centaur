
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_assign_add_inputs():
    list_of_inputs = []

    # Input 1: float32, use_locking=False
    ref = tf.Variable(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    value = tf.constant(np.array([0.5, 1.0, 1.5], dtype=np.float32))
    use_locking = False
    name = "assign_add_float32_1"
    input_dict = {"use_locking": use_locking, "name": name, "ref": ref, "value": value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int32, use_locking=True
    ref = tf.Variable(np.array([4, 5, 6], dtype=np.int32))
    value = tf.constant(np.array([-1, 0, 1], dtype=np.int32))
    use_locking = True
    name = "assign_add_int32_1"
    input_dict = {"use_locking": use_locking, "name": name, "ref": ref, "value": value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int64, use_locking=False, different shape
    ref = tf.Variable(np.array([[10, 20], [30, 40]], dtype=np.int64))
    value = tf.constant(np.array([[1, 2], [3, 4]], dtype=np.int64))
    use_locking = False
    name = "assign_add_int64_1"
    input_dict = {"use_locking": use_locking, "name": name, "ref": ref, "value": value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64, use_locking=True, scalar value
    ref = tf.Variable(np.array(5.0, dtype=np.float64))
    value = tf.constant(np.array(2.5, dtype=np.float64))
    use_locking = True
    name = "assign_add_float64_1"
    input_dict = {"use_locking": use_locking, "name": name, "ref": ref, "value": value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64
    ref = tf.Variable(np.array([1+1j, 2+2j], dtype=np.complex64))
    value = tf.constant(np.array([0.5+0.5j, 1+1j], dtype=np.complex64))
    use_locking = False
    name = "assign_add_complex64_1"
    input_dict = {"use_locking": use_locking, "name": name, "ref": ref, "value": value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: uint8
    ref = tf.Variable(np.array([10, 20, 30], dtype=np.uint8))
    value = tf.constant(np.array([5, 10, 15], dtype=np.uint8))
    use_locking = True
    name = "assign_add_uint8_1"
    input_dict = {"use_locking": use_locking, "name": name, "ref": ref, "value": value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: qint32
    ref = tf.Variable(np.array([10, 20, 30], dtype=np.int32))
    value = tf.constant(np.array([5, 10, 15], dtype=np.int32))
    use_locking = False
    name = "assign_add_qint32_1"
    input_dict = {"use_locking": use_locking, "name": name, "ref": ref, "value": value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: bfloat16
    ref = tf.Variable(np.array([1.0, 2.0], dtype=np.float16))
    value = tf.constant(np.array([0.5, 1.0], dtype=np.float16))
    use_locking = True
    name = "assign_add_bfloat16_1"
    input_dict = {"use_locking": use_locking, "name": name, "ref": ref, "value": value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: uint16, 2D array
    ref = tf.Variable(np.array([[10, 20], [30, 40]], dtype=np.uint16))
    value = tf.constant(np.array([[5, 10], [15, 20]], dtype=np.uint16))
    use_locking = False
    name = "assign_add_uint16_1"
    input_dict = {"use_locking": use_locking, "name": name, "ref": ref, "value": value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: complex128
    ref = tf.Variable(np.array([1+1j, 2+2j], dtype=np.complex128))
    value = tf.constant(np.array([0.5+0.5j, 1+1j], dtype=np.complex128))
    use_locking = True
    name = "assign_add_complex128_1"
    input_dict = {"use_locking": use_locking, "name": name, "ref": ref, "value": value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: half
    ref = tf.Variable(np.array([1.0, 2.0], dtype=np.float16))
    value = tf.constant(np.array([0.5, 1.0], dtype=np.float16))
    use_locking = False
    name = "assign_add_half_1"
    input_dict = {"use_locking": use_locking, "name": name, "ref": ref, "value": value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: uint32
    ref = tf.Variable(np.array([10, 20, 30], dtype=np.uint32))
    value = tf.constant(np.array([5, 10, 15], dtype=np.uint32))
    use_locking = True
    name = "assign_add_uint32_1"
    input_dict = {"use_locking": use_locking, "name": name, "ref": ref, "value": value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13: uint64
    ref = tf.Variable(np.array([10, 20, 30], dtype=np.uint64))
    value = tf.constant(np.array([5, 10, 15], dtype=np.uint64))
    use_locking = False
    name = "assign_add_uint64_1"
    input_dict = {"use_locking": use_locking, "name": name, "ref": ref, "value": value}
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
