
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_assign_inputs():
    list_of_inputs = []

    # Input 1
    ref = tf.Variable(np.array(1, dtype=np.int32)).handle
    value = tf.constant(np.array(2, dtype=np.int32))
    validate_shape = True
    use_locking = True
    name = "assign_op_1"

    input_dict = {
        "ref": ref,
        "value": value,
        "validate_shape": validate_shape,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    ref = tf.Variable(np.array([1, 2, 3], dtype=np.float32)).handle
    value = tf.constant(np.array([4, 5, 6], dtype=np.float32))
    validate_shape = False
    use_locking = False
    name = "assign_op_2"

    input_dict = {
        "ref": ref,
        "value": value,
        "validate_shape": validate_shape,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    ref = tf.Variable(np.array([[1, 2], [3, 4]], dtype=np.int64)).handle
    value = tf.constant(np.array([[5, 6], [7, 8]], dtype=np.int64))
    validate_shape = True
    use_locking = False
    name = "assign_op_3"

    input_dict = {
        "ref": ref,
        "value": value,
        "validate_shape": validate_shape,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    ref = tf.Variable(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64)).handle
    value = tf.constant(np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.float64))
    validate_shape = False
    use_locking = True
    name = "assign_op_4"

    input_dict = {
        "ref": ref,
        "value": value,
        "validate_shape": validate_shape,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    ref = tf.Variable(np.array([-1, -2, -3], dtype=np.int32)).handle
    value = tf.constant(np.array([-4, -5, -6], dtype=np.int32))
    validate_shape = True
    use_locking = True
    name = "assign_op_5"

    input_dict = {
        "ref": ref,
        "value": value,
        "validate_shape": validate_shape,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    ref = tf.Variable(np.array([1.5, 2.5, 3.5], dtype=np.float32)).handle
    value = tf.constant(np.array([4.5, 5.5, 6.5], dtype=np.float32))
    validate_shape = False
    use_locking = False
    name = "assign_op_6"

    input_dict = {
        "ref": ref,
        "value": value,
        "validate_shape": validate_shape,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    ref = tf.Variable(np.array([[1, 2], [3, 4]], dtype=np.int32)).handle
    value = tf.constant(np.array([[5, 6], [7, 8]], dtype=np.int32))
    validate_shape = True
    use_locking = True
    name = "assign_op_7"

    input_dict = {
        "ref": ref,
        "value": value,
        "validate_shape": validate_shape,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    ref = tf.Variable(np.array(10, dtype=np.int64)).handle
    value = tf.constant(np.array(-5, dtype=np.int64))
    validate_shape = False
    use_locking = False
    name = "assign_op_8"

    input_dict = {
        "ref": ref,
        "value": value,
        "validate_shape": validate_shape,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9
    ref = tf.Variable(np.array([1, 2, 3, 4, 5], dtype=np.float64)).handle
    value = tf.constant(np.array([6, 7, 8, 9, 10], dtype=np.float64))
    validate_shape = True
    use_locking = True
    name = "assign_op_9"

    input_dict = {
        "ref": ref,
        "value": value,
        "validate_shape": validate_shape,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    ref = tf.Variable(np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float32)).handle
    value = tf.constant(np.array([[5.5, 6.6], [7.7, 8.8]], dtype=np.float32))
    validate_shape = False
    use_locking = False
    name = "assign_op_10"

    input_dict = {
        "ref": ref,
        "value": value,
        "validate_shape": validate_shape,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Assign"] = tf_raw_ops_assign_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Assign' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Assign'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Assign', generated_inputs['tf.raw_ops.Assign'], lib="tf", suffix=0)
