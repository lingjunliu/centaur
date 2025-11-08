
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_max_inputs():
    list_of_inputs = []

    inp = np.array([1, -3, 2, 7], dtype=np.int32)
    axis = np.array(0, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"keep_dims": False, "name": "max_case_1", "input": inp, "axis": axis}))

    inp = np.array([[1.0, -2.5, 3.1], [4.2, 0.0, -7.3]], dtype=np.float32)
    axis = np.array(-1, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"keep_dims": True, "name": "max_case_2", "input": inp, "axis": axis}))

    inp = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4) - 5.5
    axis = np.array([1], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"keep_dims": False, "name": "max_case_3", "input": inp, "axis": axis}))

    inp = np.array(
        [
            [[[1.0], [2.0], [3.0]], [[-4.0], [5.0], [6.0]]],
            [[[7.5], [8.5], [9.5]], [[-10.0], [11.0], [12.0]]],
        ],
        dtype=np.float16,
    )
    axis = np.array([0, 2], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"keep_dims": True, "name": "max_case_4", "input": inp, "axis": axis}))

    inp = np.array([-128, -1, 0, 127], dtype=np.int8)
    axis = np.array(0, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"keep_dims": False, "name": "max_case_5", "input": inp, "axis": axis}))

    inp = np.array([[100, 2], [400, -5], [7, 800]], dtype=np.int16)
    axis = np.array(-2, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"keep_dims": True, "name": "max_case_6", "input": inp, "axis": axis}))

    inp = np.array(
        [
            [[1, -2], [3, 4]],
            [[-5, 6], [7, -8]],
        ],
        dtype=np.int64,
    )
    axis = np.array([1, 2], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"keep_dims": False, "name": "max_case_7", "input": inp, "axis": axis}))

    inp = np.arange(2 * 1 * 3 * 1 * 4, dtype=np.float32).reshape(2, 1, 3, 1, 4) - 10.0
    axis = np.array([0, 4], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"keep_dims": True, "name": "max_case_8", "input": inp, "axis": axis}))

    inp = np.array([[0.5], [1.5], [-2.0], [3.0]], dtype=np.float64)
    axis = np.array(0, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"keep_dims": False, "name": "max_case_9", "input": inp, "axis": axis}))

    inp = np.arange(2 * 3 * 4, dtype=np.int32).reshape(2, 3, 4) - 3
    axis = np.array(-1, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"keep_dims": True, "name": "max_case_10", "input": inp, "axis": axis}))

    return list_of_inputs

generated_inputs["tf.raw_ops.Max"] = tf_raw_ops_max_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Max' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Max'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Max', generated_inputs['tf.raw_ops.Max'], lib="tf", suffix=0)
