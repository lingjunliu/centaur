
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_quantizev2_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    min_range = np.array(-1.0, dtype=np.float32)
    max_range = np.array(1.0, dtype=np.float32)
    T = tf.qint8.as_numpy_dtype
    input_dict = {"input": input_tensor, "min_range": min_range, "max_range": max_range, "T": T, "mode": "MIN_COMBINED", "round_mode": "HALF_AWAY_FROM_ZERO", "narrow_range": False, "axis": -1, "ensure_minimum_range": 0.01, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([-5.0, -2.5, 0.0, 2.5, 5.0], dtype=np.float32)
    min_range = np.array(-5.0, dtype=np.float32)
    max_range = np.array(5.0, dtype=np.float32)
    T = tf.quint8.as_numpy_dtype
    input_dict = {"input": input_tensor, "min_range": min_range, "max_range": max_range, "T": T, "mode": "MIN_COMBINED", "round_mode": "HALF_AWAY_FROM_ZERO", "narrow_range": False, "axis": -1, "ensure_minimum_range": 0.0, "name": "test2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    min_range = np.array(0.0, dtype=np.float32)
    max_range = np.array(1.0, dtype=np.float32)
    T = tf.qint32.as_numpy_dtype
    input_dict = {"input": input_tensor, "min_range": min_range, "max_range": max_range, "T": T, "mode": "MIN_FIRST", "round_mode": "HALF_AWAY_FROM_ZERO", "narrow_range": False, "axis": -1, "ensure_minimum_range": 0.01, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    min_range = np.array(1.0, dtype=np.float32)
    max_range = np.array(5.0, dtype=np.float32)
    T = tf.qint16.as_numpy_dtype
    input_dict = {"input": input_tensor, "min_range": min_range, "max_range": max_range, "T": T, "mode": "SCALED", "round_mode": "HALF_AWAY_FROM_ZERO", "narrow_range": False, "axis": -1, "ensure_minimum_range": 0.0, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    min_range = np.array(10.0, dtype=np.float32)
    max_range = np.array(50.0, dtype=np.float32)
    T = tf.quint16.as_numpy_dtype
    input_dict = {"input": input_tensor, "min_range": min_range, "max_range": max_range, "T": T, "mode": "SCALED", "round_mode": "HALF_AWAY_FROM_ZERO", "narrow_range": True, "axis": -1, "ensure_minimum_range": 0.01, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[-1.0, -0.5], [0.0, 0.5], [1.0, 1.5]], dtype=np.float32)
    min_range = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    max_range = np.array([1.0, 1.5, 2.0], dtype=np.float32)
    T = tf.qint8.as_numpy_dtype
    input_dict = {"input": input_tensor, "min_range": min_range, "max_range": max_range, "T": T, "mode": "MIN_COMBINED", "round_mode": "HALF_AWAY_FROM_ZERO", "narrow_range": False, "axis": 0, "ensure_minimum_range": 0.0, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    min_range = np.array([1.0, 3.0], dtype=np.float32)
    max_range = np.array([2.0, 4.0], dtype=np.float32)
    T = tf.quint8.as_numpy_dtype
    input_dict = {"input": input_tensor, "min_range": min_range, "max_range": max_range, "T": T, "mode": "MIN_COMBINED", "round_mode": "HALF_AWAY_FROM_ZERO", "narrow_range": False, "axis": 1, "ensure_minimum_range": 0.01, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    min_range = np.array(0.5, dtype=np.float32)
    max_range = np.array(3.5, dtype=np.float32)
    T = tf.qint16.as_numpy_dtype
    input_dict = {"input": input_tensor, "min_range": min_range, "max_range": max_range, "T": T, "mode": "SCALED", "round_mode": "HALF_AWAY_FROM_ZERO", "narrow_range": False, "axis": -1, "ensure_minimum_range": 0.1, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9
    input_tensor = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    min_range = np.array(-1.0, dtype=np.float32)
    max_range = np.array(1.0, dtype=np.float32)
    T = tf.qint16.as_numpy_dtype
    input_dict = {"input": input_tensor, "min_range": min_range, "max_range": max_range, "T": T, "mode": "MIN_COMBINED", "round_mode": "HALF_AWAY_FROM_ZERO", "narrow_range": False, "axis": -1, "ensure_minimum_range": 0.001, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    min_range = np.array(0.0, dtype=np.float32)
    max_range = np.array(0.5, dtype=np.float32)
    T = tf.quint16.as_numpy_dtype
    input_dict = {"input": input_tensor, "min_range": min_range, "max_range": max_range, "T": T, "mode": "MIN_FIRST", "round_mode": "HALF_AWAY_FROM_ZERO", "narrow_range": True, "axis": -1, "ensure_minimum_range": 0.0, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    input_tensor = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    min_range = np.array(1.0, dtype=np.float32)
    max_range = np.array(5.0, dtype=np.float32)
    T = tf.qint8.as_numpy_dtype
    input_dict = {"input": input_tensor, "min_range": min_range, "max_range": max_range, "T": T, "mode": "SCALED", "round_mode": "HALF_AWAY_FROM_ZERO", "narrow_range": True, "axis": -1, "ensure_minimum_range": 0.0, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12
    input_tensor = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    min_range = np.array(-1.0, dtype=np.float32)
    max_range = np.array(1.0, dtype=np.float32)
    T = tf.qint8.as_numpy_dtype
    input_dict = {"input": input_tensor, "min_range": min_range, "max_range": max_range, "T": T, "mode": "MIN_COMBINED", "round_mode": "HALF_AWAY_FROM_ZERO", "narrow_range": False, "axis": -1, "ensure_minimum_range": 0.01, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 13
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    min_range = np.array(0.5, dtype=np.float32)
    max_range = np.array(3.5, dtype=np.float32)
    T = tf.qint16.as_numpy_dtype
    input_dict = {"input": input_tensor, "min_range": min_range, "max_range": max_range, "T": T, "mode": "SCALED", "round_mode": "HALF_AWAY_FROM_ZERO", "narrow_range": False, "axis": -1, "ensure_minimum_range": 0.1, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 14
    input_tensor = np.array([[-1.0, -0.5], [0.0, 0.5], [1.0, 1.5]], dtype=np.float32)
    min_range = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    max_range = np.array([1.0, 1.5, 2.0], dtype=np.float32)
    T = tf.qint8.as_numpy_dtype
    input_dict = {"input": input_tensor, "min_range": min_range, "max_range": max_range, "T": T, "mode": "MIN_COMBINED", "round_mode": "HALF_AWAY_FROM_ZERO", "narrow_range": False, "axis": 0, "ensure_minimum_range": 0.0, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 15
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    min_range = np.array([1.0, 3.0], dtype=np.float32)
    max_range = np.array([2.0, 4.0], dtype=np.float32)
    T = tf.quint8.as_numpy_dtype
    input_dict = {"input": input_tensor, "min_range": min_range, "max_range": max_range, "T": T, "mode": "MIN_COMBINED", "round_mode": "HALF_AWAY_FROM_ZERO", "narrow_range": False, "axis": 1, "ensure_minimum_range": 0.01, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 16
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    min_range = np.array([1.0, 5.0], dtype=np.float32)
    max_range = np.array([4.0, 8.0], dtype=np.float32)
    T = tf.quint8.as_numpy_dtype
    input_dict = {"input": input_tensor, "min_range": min_range, "max_range": max_range, "T": T, "mode": "MIN_COMBINED", "round_mode": "HALF_AWAY_FROM_ZERO", "narrow_range": False, "axis": 0, "ensure_minimum_range": 0.01, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QuantizeV2"] = tf_raw_ops_quantizev2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.QuantizeV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizeV2'.")

check_valid('tf.raw_ops.QuantizeV2', generated_inputs['tf.raw_ops.QuantizeV2'], lib="tf", suffix=0)
