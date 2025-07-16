
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_assign_add_inputs():
    list_of_inputs = []

    # Input 1: Simple float32 addition
    ref = tf.Variable(np.array(1.0, dtype=np.float32))
    value = tf.constant(np.array(2.0, dtype=np.float32))
    use_locking = False
    name = "float_add_1"
    input_dict = {"ref": ref, "value": value, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int32 addition with locking
    ref = tf.Variable(np.array(5, dtype=np.int32))
    value = tf.constant(np.array(3, dtype=np.int32))
    use_locking = True
    name = "int_add_1"
    input_dict = {"ref": ref, "value": value, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64 addition with multi-dimensional array
    ref = tf.Variable(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
    value = tf.constant(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64))
    use_locking = False
    name = "float_add_2"
    input_dict = {"ref": ref, "value": value, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int64 addition with negative value
    ref = tf.Variable(np.array(10, dtype=np.int64))
    value = tf.constant(np.array(-5, dtype=np.int64))
    use_locking = False
    name = "int_add_neg"
    input_dict = {"ref": ref, "value": value, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64 addition
    ref = tf.Variable(np.array(1 + 2j, dtype=np.complex64))
    value = tf.constant(np.array(3 + 4j, dtype=np.complex64))
    use_locking = False
    name = "complex_add"
    input_dict = {"ref": ref, "value": value, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: uint8 addition
    ref = tf.Variable(np.array(200, dtype=np.uint8))
    value = tf.constant(np.array(50, dtype=np.uint8))
    use_locking = False
    name = "uint8_add"
    input_dict = {"ref": ref, "value": value, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: bfloat16 addition
    ref = tf.Variable(np.array(1.0, dtype=np.bfloat16))
    value = tf.constant(np.array(2.0, dtype=np.bfloat16))
    use_locking = False
    name = "bfloat16_add"
    input_dict = {"ref": ref, "value": value, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Multiple dimensions for int32
    ref = tf.Variable(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32))
    value = tf.constant(np.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]]], dtype=np.int32))
    use_locking = False
    name = "int_add_multi"
    input_dict = {"ref": ref, "value": value, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9:  uint16 addition
    ref = tf.Variable(np.array(60000, dtype=np.uint16))
    value = tf.constant(np.array(5000, dtype=np.uint16))
    use_locking = False
    name = "uint16_add"
    input_dict = {"ref": ref, "value": value, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Half type
    ref = tf.Variable(np.array(1.0, dtype=np.float16))
    value = tf.constant(np.array(2.0, dtype=np.float16))
    use_locking = False
    name = "half_add"
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
