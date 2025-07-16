
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_assign_sub_inputs():
    list_of_inputs = []

    # Input 1: Basic example with float32
    ref = tf.Variable(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    value = tf.constant(np.array([0.5, 1.0, 1.5], dtype=np.float32) , dtype=tf.float32)
    use_locking = False
    name = "assign_sub_1"

    input_dict = {
        "ref": ref,
        "value": value,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int32 with locking
    ref = tf.Variable(np.array([10, 20, 30], dtype=np.int32))
    value = tf.constant(np.array([3, 7, 11], dtype=np.int32) , dtype=tf.int32)
    use_locking = True
    name = "assign_sub_2"

    input_dict = {
        "ref": ref,
        "value": value,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64 with negative values
    ref = tf.Variable(np.array([1.0, 2.0, 3.0], dtype=np.float64))
    value = tf.constant(np.array([-0.5, -1.0, -1.5], dtype=np.float64) , dtype=tf.float64)
    use_locking = False
    name = "assign_sub_3"

    input_dict = {
        "ref": ref,
        "value": value,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array with int64
    ref = tf.Variable(np.array([[10, 20], [30, 40]], dtype=np.int64))
    value = tf.constant(np.array([[1, 2], [3, 4]], dtype=np.int64) , dtype=tf.int64)
    use_locking = True
    name = "assign_sub_4"

    input_dict = {
        "ref": ref,
        "value": value,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64
    ref = tf.Variable(np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64))
    value = tf.constant(np.array([0.5+0.5j, 1+1j, 1.5+1.5j], dtype=np.complex64), dtype=tf.complex64)
    use_locking = False
    name = "assign_sub_5"

    input_dict = {
        "ref": ref,
        "value": value,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: uint8
    ref = tf.Variable(np.array([255, 200, 150], dtype=np.uint8))
    value = tf.constant(np.array([10, 20, 30], dtype=np.uint8), dtype=tf.uint8)
    use_locking = True
    name = "assign_sub_6"

    input_dict = {
        "ref": ref,
        "value": value,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: bfloat16 (needs casting)
    ref = tf.Variable(tf.cast(np.array([1.0, 2.0, 3.0], dtype=np.float32), dtype=tf.bfloat16))
    value = tf.constant(tf.cast(np.array([0.5, 1.0, 1.5], dtype=np.float32), dtype=tf.bfloat16), dtype=tf.bfloat16)
    use_locking = False
    name = "assign_sub_7"

    input_dict = {
        "ref": ref,
        "value": value,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8:  rank 3 tensor with int16
    ref = tf.Variable(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int16))
    value = tf.constant(np.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]]], dtype=np.int16), dtype=tf.int16)
    use_locking = True
    name = "assign_sub_8"

    input_dict = {
        "ref": ref,
        "value": value,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: uint32
    ref = tf.Variable(np.array([1000, 2000, 3000], dtype=np.uint32))
    value = tf.constant(np.array([100, 200, 300], dtype=np.uint32), dtype=tf.uint32)
    use_locking = False
    name = "assign_sub_9"

    input_dict = {
        "ref": ref,
        "value": value,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: uint64
    ref = tf.Variable(np.array([1000, 2000, 3000], dtype=np.uint64))
    value = tf.constant(np.array([100, 200, 300], dtype=np.uint64), dtype=tf.uint64)
    use_locking = True
    name = "assign_sub_10"

    input_dict = {
        "ref": ref,
        "value": value,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
temp_inputs = tf_raw_ops_assign_sub_inputs()
generated_inputs["tf.raw_ops.AssignSub"] = temp_inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.AssignSub' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AssignSub'.")

check_valid('tf.raw_ops.AssignSub', generated_inputs['tf.raw_ops.AssignSub'], lib="tf", suffix=0)
