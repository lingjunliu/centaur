
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_nest_assert_same_structure_inputs():
    list_of_inputs = []

    nest1 = [np.int32(1), np.int32(2), np.int32(3)]
    nest2 = [np.int64(9), np.int16(-2), np.int8(0)]
    list_of_inputs.append(copy.deepcopy({"nest1": nest1, "nest2": nest2, "check_types": True, "expand_composites": False}))

    nest1 = [np.array([[1.0, -2.5], [3.3, 4.4]], dtype=np.float32),
             np.array([[5.5, 6.6], [7.7, 8.8]], dtype=np.float32)]
    nest2 = [np.array([[0.0, 2.5], [-3.3, -4.4]], dtype=np.float64),
             np.array([[9.9, -6.6], [7.7, -8.8]], dtype=np.float64)]
    list_of_inputs.append(copy.deepcopy({"nest1": nest1, "nest2": nest2, "check_types": True, "expand_composites": False}))

    nest1 = [[np.int32(1), np.int32(2)], [np.int32(3), np.int32(4)]]
    nest2 = [[np.int64(10), np.int64(20)], [np.int64(30), np.int64(40)]]
    list_of_inputs.append(copy.deepcopy({"nest1": nest1, "nest2": nest2, "check_types": False, "expand_composites": False}))

    nest1 = [np.bool_(True), np.bool_(False), np.bool_(True), np.bool_(False)]
    nest2 = [np.bool_(False), np.bool_(True), np.bool_(False), np.bool_(True)]
    list_of_inputs.append(copy.deepcopy({"nest1": nest1, "nest2": nest2, "check_types": True, "expand_composites": False}))

    nest1 = [np.array([1.0, -2.0, 3.5], dtype=np.float32),
             np.array([-4.5, 5.0, -6.1], dtype=np.float32)]
    nest2 = [np.array([0.0, 2.0, -3.5], dtype=np.float32),
             np.array([4.5, -5.0, 6.1], dtype=np.float32)]
    list_of_inputs.append(copy.deepcopy({"nest1": nest1, "nest2": nest2, "check_types": False, "expand_composites": True}))

    nest1 = [[np.bool_(True), np.bool_(True), np.bool_(False)],
             [np.bool_(False), np.bool_(True), np.bool_(False)]]
    nest2 = [[np.bool_(False), np.bool_(False), np.bool_(True)],
             [np.bool_(True), np.bool_(False), np.bool_(True)]]
    list_of_inputs.append(copy.deepcopy({"nest1": nest1, "nest2": nest2, "check_types": True, "expand_composites": True}))

    nest1 = [np.array([[[1, 2, 3]], [[4, 5, 6]]], dtype=np.int32),
             np.array([[[7, 8, 9]], [[10, 11, 12]]], dtype=np.int32)]
    nest2 = [np.array([[[0, -2, -3]], [[-4, -5, -6]]], dtype=np.int64),
             np.array([[[7, -8, 9]], [[-10, 11, -12]]], dtype=np.int64)]
    list_of_inputs.append(copy.deepcopy({"nest1": nest1, "nest2": nest2, "check_types": False, "expand_composites": False}))

    nest1 = [[np.array([1, 2], dtype=np.int32), np.array([3, 4], dtype=np.int32)],
             [np.array([5, 6], dtype=np.int32), np.array([7, 8], dtype=np.int32)]]
    nest2 = [[np.array([-1, -2], dtype=np.int32), np.array([-3, -4], dtype=np.int32)],
             [np.array([-5, -6], dtype=np.int32), np.array([-7, -8], dtype=np.int32)]]
    list_of_inputs.append(copy.deepcopy({"nest1": nest1, "nest2": nest2, "check_types": True, "expand_composites": False}))

    nest1 = [np.array([10, 20, 30], dtype=np.int16),
             np.array([40, 50, 60], dtype=np.int16),
             np.array([70, 80, 90], dtype=np.int16)]
    nest2 = [np.array([-10, -20, -30], dtype=np.int32),
             np.array([-40, -50, -60], dtype=np.int32),
             np.array([-70, -80, -90], dtype=np.int32)]
    list_of_inputs.append(copy.deepcopy({"nest1": nest1, "nest2": nest2, "check_types": False, "expand_composites": True}))

    nest1 = [np.float64(np.nan), np.float64(np.inf), np.float64(-np.inf)]
    nest2 = [np.float32(0.0), np.float32(-1e9), np.float32(1e9)]
    list_of_inputs.append(copy.deepcopy({"nest1": nest1, "nest2": nest2, "check_types": True, "expand_composites": False}))

    nest1 = [np.array([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12],
                       [13, 14, 15, 16]], dtype=np.int32),
             np.array([[16, 15, 14, 13],
                       [12, 11, 10, 9],
                       [8, 7, 6, 5],
                       [4, 3, 2, 1]], dtype=np.int32)]
    nest2 = [np.array([[0, -2, -3, -4],
                       [-5, -6, -7, -8],
                       [-9, -10, -11, -12],
                       [-13, -14, -15, -16]], dtype=np.int64),
             np.array([[6, 5, 4, 3],
                       [2, 1, 0, -1],
                       [-2, -3, -4, -5],
                       [-6, -7, -8, -9]], dtype=np.int64)]
    list_of_inputs.append(copy.deepcopy({"nest1": nest1, "nest2": nest2, "check_types": True, "expand_composites": True}))

    nest1 = [[np.float32(1.1), np.float32(-2.2)],
             [np.float32(3.3), np.float32(-4.4)],
             [np.float32(5.5), np.float32(-6.6)]]
    nest2 = [[np.float64(-1.1), np.float64(2.2)],
             [np.float64(-3.3), np.float64(4.4)],
             [np.float64(-5.5), np.float64(6.6)]]
    list_of_inputs.append(copy.deepcopy({"nest1": nest1, "nest2": nest2, "check_types": False, "expand_composites": False}))

    return list_of_inputs

generated_inputs["tf.nest.assert_same_structure"] = tf_nest_assert_same_structure_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nest.assert_same_structure' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nest.assert_same_structure'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nest.assert_same_structure', generated_inputs['tf.nest.assert_same_structure'], lib="tf", suffix=0)
