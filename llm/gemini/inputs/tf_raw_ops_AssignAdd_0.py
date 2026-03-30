
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_assign_add_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D
    ref = tf.Variable(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    value = tf.constant(np.array([0.5, 1.0, 1.5], dtype=np.float32))
    input_dict = {"ref": ref, "value": value, "use_locking": False, "name": "add1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int32, 2D
    ref = tf.Variable(np.array([[1, 2], [3, 4]], dtype=np.int32))
    value = tf.constant(np.array([[5, 6], [7, 8]], dtype=np.int32))
    input_dict = {"ref": ref, "value": value, "use_locking": True, "name": "add2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64, 0D
    ref = tf.Variable(np.array(10.0, dtype=np.float64))
    value = tf.constant(np.array(5.0, dtype=np.float64))
    input_dict = {"ref": ref, "value": value, "use_locking": False, "name": "add3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int64, 3D
    ref = tf.Variable(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64))
    value = tf.constant(np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.int64))
    input_dict = {"ref": ref, "value": value, "use_locking": True, "name": "add4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: uint8, 1D
    ref = tf.Variable(np.array([100, 200, 250], dtype=np.uint8))
    value = tf.constant(np.array([10, 20, 5], dtype=np.uint8))
    input_dict = {"ref": ref, "value": value, "use_locking": False, "name": "add5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int16, 2D, negative values
    ref = tf.Variable(np.array([[-1, -2], [-3, -4]], dtype=np.int16))
    value = tf.constant(np.array([[5, 6], [7, 8]], dtype=np.int16))
    input_dict = {"ref": ref, "value": value, "use_locking": True, "name": "add6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: complex64, 1D
    ref = tf.Variable(np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64))
    value = tf.constant(np.array([0.5+0.5j, 1.0+1.0j, 1.5+1.5j], dtype=np.complex64))
    input_dict = {"ref": ref, "value": value, "use_locking": False, "name": "add7"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.AssignAdd"] = tf_raw_ops_assign_add_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.AssignAdd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AssignAdd'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.AssignAdd', generated_inputs['tf.raw_ops.AssignAdd'], lib="tf", suffix=0)
