
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_assign_sub_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D
    ref = tf.Variable(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    value = tf.constant(np.array([0.5, 1.0, 1.5], dtype=np.float32))
    input_dict = {"ref": ref, "value": value, "use_locking": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int32, 2D
    ref = tf.Variable(np.array([[1, 2], [3, 4]], dtype=np.int32))
    value = tf.constant(np.array([[0, 1], [2, 1]], dtype=np.int32))
    input_dict = {"ref": ref, "value": value, "use_locking": True, "name": "int32_2d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64, 1D, negative values
    ref = tf.Variable(np.array([-1.0, 2.0, -3.0], dtype=np.float64))
    value = tf.constant(np.array([0.5, -1.0, 1.5], dtype=np.float64))
    input_dict = {"ref": ref, "value": value, "use_locking": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int64, 3D
    ref = tf.Variable(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64))
    value = tf.constant(np.array([[[0, 1], [2, 1]], [[1, 0], [0, 1]]], dtype=np.int64))
    input_dict = {"ref": ref, "value": value, "use_locking": True, "name": "int64_3d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: uint8, 1D
    ref = tf.Variable(np.array([255, 128, 0], dtype=np.uint8))
    value = tf.constant(np.array([10, 20, 30], dtype=np.uint8))
    input_dict = {"ref": ref, "value": value, "use_locking": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: complex64, 1D
    ref = tf.Variable(np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64))
    value = tf.constant(np.array([0.5+0.5j, 1+1j, 1.5+1.5j], dtype=np.complex64))
    input_dict = {"ref": ref, "value": value, "use_locking": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: uint32, 1D
    ref = tf.Variable(np.array([1000, 2000, 3000], dtype=np.uint32))
    value = tf.constant(np.array([100, 200, 300], dtype=np.uint32))
    input_dict = {"ref": ref, "value": value, "use_locking": True, "name": "uint32_1d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: int16, 1D
    ref = tf.Variable(np.array([10, 20, 30], dtype=np.int16))
    value = tf.constant(np.array([1, 2, 3], dtype=np.int16))
    input_dict = {"ref": ref, "value": value, "use_locking": False, "name": "int16_1d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: uint16, 1D.
    ref = tf.Variable(np.array([200, 150, 100], dtype=np.uint16))
    value = tf.constant(np.array([10, 20, 30], dtype=np.uint16))
    input_dict = {"ref": ref, "value": value, "use_locking": True, "name": "uint16_1d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: int8, 2D
    ref = tf.Variable(np.array([[1, 2], [3, 4]], dtype=np.int8))
    value = tf.constant(np.array([[0, 1], [2, 1]], dtype=np.int8))
    input_dict = {"ref": ref, "value": value, "use_locking": True, "name": "int8_2d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: complex128, 1D
    ref = tf.Variable(np.array([1+1j, 2+2j, 3+3j], dtype=np.complex128))
    value = tf.constant(np.array([0.5+0.5j, 1+1j, 1.5+1.5j], dtype=np.complex128))
    input_dict = {"ref": ref, "value": value, "use_locking": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.AssignSub"] = tf_raw_ops_assign_sub_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.AssignSub' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AssignSub'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.AssignSub', generated_inputs['tf.raw_ops.AssignSub'], lib="tf", suffix=0)
