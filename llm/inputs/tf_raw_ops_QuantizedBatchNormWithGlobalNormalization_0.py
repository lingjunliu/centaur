
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def quantized_batch_norm_with_global_normalization_inputs():
    list_of_inputs = []

    # Input 1
    t = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.int8)
    t_min = np.array(-1.0, dtype=np.float32)
    t_max = np.array(1.0, dtype=np.float32)
    m = np.array([0.5, 0.6], dtype=np.int8)
    m_min = np.array(-0.5, dtype=np.float32)
    m_max = np.array(0.5, dtype=np.float32)
    v = np.array([0.1, 0.2], dtype=np.int8)
    v_min = np.array(0.0, dtype=np.float32)
    v_max = np.array(0.3, dtype=np.float32)
    beta = np.array([0.0, 0.1], dtype=np.int8)
    beta_min = np.array(-0.1, dtype=np.float32)
    beta_max = np.array(0.2, dtype=np.float32)
    gamma = np.array([1.0, 1.1], dtype=np.int8)
    gamma_min = np.array(0.9, dtype=np.float32)
    gamma_max = np.array(1.2, dtype=np.float32)
    out_type = tf.qint8
    variance_epsilon = 0.001
    scale_after_normalization = True
    name = "batch_norm_1"

    input_dict = {
        "t": t,
        "t_min": t_min,
        "t_max": t_max,
        "m": m,
        "m_min": m_min,
        "m_max": m_max,
        "v": v,
        "v_min": v_min,
        "v_max": v_max,
        "beta": beta,
        "beta_min": beta_min,
        "beta_max": beta_max,
        "gamma": gamma,
        "gamma_min": gamma_min,
        "gamma_max": gamma_max,
        "out_type": out_type,
        "variance_epsilon": variance_epsilon,
        "scale_after_normalization": scale_after_normalization,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    t = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.uint8)
    t_min = np.array(0.0, dtype=np.float32)
    t_max = np.array(255.0, dtype=np.float32)
    m = np.array([100, 120], dtype=np.uint8)
    m_min = np.array(50.0, dtype=np.float32)
    m_max = np.array(150.0, dtype=np.float32)
    v = np.array([20, 30], dtype=np.uint8)
    v_min = np.array(10.0, dtype=np.float32)
    v_max = np.array(40.0, dtype=np.float32)
    beta = np.array([5, 10], dtype=np.uint8)
    beta_min = np.array(0.0, dtype=np.float32)
    beta_max = np.array(15.0, dtype=np.float32)
    gamma = np.array([2, 3], dtype=np.uint8)
    gamma_min = np.array(1.0, dtype=np.float32)
    gamma_max = np.array(4.0, dtype=np.float32)
    out_type = tf.quint8
    variance_epsilon = 0.01
    scale_after_normalization = False
    name = "batch_norm_2"

    input_dict = {
        "t": t,
        "t_min": t_min,
        "t_max": t_max,
        "m": m,
        "m_min": m_min,
        "m_max": m_max,
        "v": v,
        "v_min": v_min,
        "v_max": v_max,
        "beta": beta,
        "beta_min": beta_min,
        "beta_max": beta_max,
        "gamma": gamma,
        "gamma_min": gamma_min,
        "gamma_max": gamma_max,
        "out_type": out_type,
        "variance_epsilon": variance_epsilon,
        "scale_after_normalization": scale_after_normalization,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    t = np.array([[[[10, 20], [30, 40]]]], dtype=np.int32)
    t_min = np.array(-100.0, dtype=np.float32)
    t_max = np.array(100.0, dtype=np.float32)
    m = np.array([5, 6], dtype=np.int32)
    m_min = np.array(-10.0, dtype=np.float32)
    m_max = np.array(10.0, dtype=np.float32)
    v = np.array([1, 2], dtype=np.int32)
    v_min = np.array(0.0, dtype=np.float32)
    v_max = np.array(5.0, dtype=np.float32)
    beta = np.array([0, 1], dtype=np.int32)
    beta_min = np.array(-2.0, dtype=np.float32)
    beta_max = np.array(2.0, dtype=np.float32)
    gamma = np.array([2, 3], dtype=np.int32)
    gamma_min = np.array(1.0, dtype=np.float32)
    gamma_max = np.array(4.0, dtype=np.float32)
    out_type = tf.qint32
    variance_epsilon = 0.1
    scale_after_normalization = True
    name = "batch_norm_3"

    input_dict = {
        "t": t,
        "t_min": t_min,
        "t_max": t_max,
        "m": m,
        "m_min": m_min,
        "m_max": m_max,
        "v": v,
        "v_min": v_min,
        "v_max": v_max,
        "beta": beta,
        "beta_min": beta_min,
        "beta_max": beta_max,
        "gamma": gamma,
        "gamma_min": gamma_min,
        "gamma_max": gamma_max,
        "out_type": out_type,
        "variance_epsilon": variance_epsilon,
        "scale_after_normalization": scale_after_normalization,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    t = np.array([[[[100, 200]]]], dtype=np.int16)
    t_min = np.array(-200.0, dtype=np.float32)
    t_max = np.array(200.0, dtype=np.float32)
    m = np.array([50], dtype=np.int16)
    m_min = np.array(-100.0, dtype=np.float32)
    m_max = np.array(100.0, dtype=np.float32)
    v = np.array([10], dtype=np.int16)
    v_min = np.array(0.0, dtype=np.float32)
    v_max = np.array(20.0, dtype=np.float32)
    beta = np.array([0], dtype=np.int16)
    beta_min = np.array(-10.0, dtype=np.float32)
    beta_max = np.array(10.0, dtype=np.float32)
    gamma = np.array([2], dtype=np.int16)
    gamma_min = np.array(0.0, dtype=np.float32)
    gamma_max = np.array(5.0, dtype=np.float32)
    out_type = tf.qint16
    variance_epsilon = 0.0001
    scale_after_normalization = False
    name = "batch_norm_4"

    input_dict = {
        "t": t,
        "t_min": t_min,
        "t_max": t_max,
        "m": m,
        "m_min": m_min,
        "m_max": m_max,
        "v": v,
        "v_min": v_min,
        "v_max": v_max,
        "beta": beta,
        "beta_min": beta_min,
        "beta_max": beta_max,
        "gamma": gamma,
        "gamma_min": gamma_min,
        "gamma_max": gamma_max,
        "out_type": out_type,
        "variance_epsilon": variance_epsilon,
        "scale_after_normalization": scale_after_normalization,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    t = np.array([[[[500, 600]]]], dtype=np.uint16)
    t_min = np.array(0.0, dtype=np.float32)
    t_max = np.array(1000.0, dtype=np.float32)
    m = np.array([300], dtype=np.uint16)
    m_min = np.array(0.0, dtype=np.float32)
    m_max = np.array(500.0, dtype=np.float32)
    v = np.array([50], dtype=np.uint16)
    v_min = np.array(0.0, dtype=np.float32)
    v_max = np.array(100.0, dtype=np.float32)
    beta = np.array([20], dtype=np.uint16)
    beta_min = np.array(0.0, dtype=np.float32)
    beta_max = np.array(50.0, dtype=np.float32)
    gamma = np.array([5], dtype=np.uint16)
    gamma_min = np.array(0.0, dtype=np.float32)
    gamma_max = np.array(10.0, dtype=np.float32)
    out_type = tf.quint16
    variance_epsilon = 0.00001
    scale_after_normalization = True
    name = "batch_norm_5"

    input_dict = {
        "t": t,
        "t_min": t_min,
        "t_max": t_max,
        "m": m,
        "m_min": m_min,
        "m_max": m_max,
        "v": v,
        "v_min": v_min,
        "v_max": v_max,
        "beta": beta,
        "beta_min": beta_min,
        "beta_max": beta_max,
        "gamma": gamma,
        "gamma_min": gamma_min,
        "gamma_max": gamma_max,
        "out_type": out_type,
        "variance_epsilon": variance_epsilon,
        "scale_after_normalization": scale_after_normalization,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    t = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]], dtype=np.int8)
    t_min = np.array(-1.0, dtype=np.float32)
    t_max = np.array(1.0, dtype=np.float32)
    m = np.array([0.5, 0.6, 0.7], dtype=np.int8)
    m_min = np.array(-0.5, dtype=np.float32)
    m_max = np.array(0.5, dtype=np.float32)
    v = np.array([0.1, 0.2, 0.3], dtype=np.int8)
    v_min = np.array(0.0, dtype=np.float32)
    v_max = np.array(0.3, dtype=np.float32)
    beta = np.array([0.0, 0.1, 0.2], dtype=np.int8)
    beta_min = np.array(-0.1, dtype=np.float32)
    beta_max = np.array(0.2, dtype=np.float32)
    gamma = np.array([1.0, 1.1, 1.2], dtype=np.int8)
    gamma_min = np.array(0.9, dtype=np.float32)
    gamma_max = np.array(1.2, dtype=np.float32)
    out_type = tf.qint8
    variance_epsilon = 0.001
    scale_after_normalization = True
    name = "batch_norm_6"

    input_dict = {
        "t": t,
        "t_min": t_min,
        "t_max": t_max,
        "m": m,
        "m_min": m_min,
        "m_max": m_max,
        "v": v,
        "v_min": v_min,
        "v_max": v_max,
        "beta": beta,
        "beta_min": beta_min,
        "beta_max": beta_max,
        "gamma": gamma,
        "gamma_min": gamma_min,
        "gamma_max": gamma_max,
        "out_type": out_type,
        "variance_epsilon": variance_epsilon,
        "scale_after_normalization": scale_after_normalization,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    t = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.int8)
    t_min = np.array(-5.0, dtype=np.float32)
    t_max = np.array(5.0, dtype=np.float32)
    m = np.array([0,0], dtype=np.int8)
    m_min = np.array(-1.0, dtype=np.float32)
    m_max = np.array(1.0, dtype=np.float32)
    v = np.array([2,2], dtype=np.int8)
    v_min = np.array(0.0, dtype=np.float32)
    v_max = np.array(4.0, dtype=np.float32)
    beta = np.array([1,1], dtype=np.int8)
    beta_min = np.array(-2.0, dtype=np.float32)
    beta_max = np.array(2.0, dtype=np.float32)
    gamma = np.array([1,1], dtype=np.int8)
    gamma_min = np.array(-2.0, dtype=np.float32)
    gamma_max = np.array(2.0, dtype=np.float32)
    out_type = tf.qint8
    variance_epsilon = 1e-6
    scale_after_normalization = True
    name = "batch_norm_7"

    input_dict = {
        "t": t,
        "t_min": t_min,
        "t_max": t_max,
        "m": m,
        "m_min": m_min,
        "m_max": m_max,
        "v": v,
        "v_min": v_min,
        "v_max": v_max,
        "beta": beta,
        "beta_min": beta_min,
        "beta_max": beta_max,
        "gamma": gamma,
        "gamma_min": gamma_min,
        "gamma_max": gamma_max,
        "out_type": out_type,
        "variance_epsilon": variance_epsilon,
        "scale_after_normalization": scale_after_normalization,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    t = np.array([[[[1, 2], [3, 4]]]], dtype=np.int8)
    t_min = np.array(-5.0, dtype=np.float32)
    t_max = np.array(5.0, dtype=np.float32)
    m = np.array([0,0], dtype=np.int8)
    m_min = np.array(-1.0, dtype=np.float32)
    m_max = np.array(1.0, dtype=np.float32)
    v = np.array([2,2], dtype=np.int8)
    v_min = np.array(0.0, dtype=np.float32)
    v_max = np.array(4.0, dtype=np.float32)
    beta = np.array([1,1], dtype=np.int8)
    beta_min = np.array(-2.0, dtype=np.float32)
    beta_max = np.array(2.0, dtype=np.float32)
    gamma = np.array([1,1], dtype=np.int8)
    gamma_min = np.array(-2.0, dtype=np.float32)
    gamma_max = np.array(2.0, dtype=np.float32)
    out_type = tf.qint8
    variance_epsilon = 1e-6
    scale_after_normalization = True
    name = "batch_norm_8"

    input_dict = {
        "t": t,
        "t_min": t_min,
        "t_max": t_max,
        "m": m,
        "m_min": m_min,
        "m_max": m_max,
        "v": v,
        "v_min": v_min,
        "v_max": v_max,
        "beta": beta,
        "beta_min": beta_min,
        "beta_max": beta_max,
        "gamma": gamma,
        "gamma_min": gamma_min,
        "gamma_max": gamma_max,
        "out_type": out_type,
        "variance_epsilon": variance_epsilon,
        "scale_after_normalization": scale_after_normalization,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    t = np.array([[[[1, 2], [3, 4]]]], dtype=np.int8)
    t_min = np.array(-5.0, dtype=np.float32)
    t_max = np.array(5.0, dtype=np.float32)
    m = np.array([0,0], dtype=np.int8)
    m_min = np.array(-1.0, dtype=np.float32)
    m_max = np.array(1.0, dtype=np.float32)
    v = np.array([2,2], dtype=np.int8)
    v_min = np.array(0.0, dtype=np.float32)
    v_max = np.array(4.0, dtype=np.float32)
    beta = np.array([1,1], dtype=np.int8)
    beta_min = np.array(-2.0, dtype=np.float32)
    beta_max = np.array(2.0, dtype=np.float32)
    gamma = np.array([1,1], dtype=np.int8)
    gamma_min = np.array(-2.0, dtype=np.float32)
    gamma_max = np.array(2.0, dtype=np.float32)
    out_type = tf.qint8
    variance_epsilon = 1e-6
    scale_after_normalization = False
    name = "batch_norm_9"

    input_dict = {
        "t": t,
        "t_min": t_min,
        "t_max": t_max,
        "m": m,
        "m_min": m_min,
        "m_max": m_max,
        "v": v,
        "v_min": v_min,
        "v_max": v_max,
        "beta": beta,
        "beta_min": beta_min,
        "beta_max": beta_max,
        "gamma": gamma,
        "gamma_min": gamma_min,
        "gamma_max": gamma_max,
        "out_type": out_type,
        "variance_epsilon": variance_epsilon,
        "scale_after_normalization": scale_after_normalization,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    t = np.array([[[[1, 2], [3, 4]]]], dtype=np.int32)
    t_min = np.array(-5.0, dtype=np.float32)
    t_max = np.array(5.0, dtype=np.float32)
    m = np.array([0,0], dtype=np.int32)
    m_min = np.array(-1.0, dtype=np.float32)
    m_max = np.array(1.0, dtype=np.float32)
    v = np.array([2,2], dtype=np.int32)
    v_min = np.array(0.0, dtype=np.float32)
    v_max = np.array(4.0, dtype=np.float32)
    beta = np.array([1,1], dtype=np.int32)
    beta_min = np.array(-2.0, dtype=np.float32)
    beta_max = np.array(2.0, dtype=np.float32)
    gamma = np.array([1,1], dtype=np.int32)
    gamma_min = np.array(-2.0, dtype=np.float32)
    gamma_max = np.array(2.0, dtype=np.float32)
    out_type = tf.qint32
    variance_epsilon = 1e-6
    scale_after_normalization = False
    name = "batch_norm_10"

    input_dict = {
        "t": t,
        "t_min": t_min,
        "t_max": t_max,
        "m": m,
        "m_min": m_min,
        "m_max": m_max,
        "v": v,
        "v_min": v_min,
        "v_max": v_max,
        "beta": beta,
        "beta_min": beta_min,
        "beta_max": beta_max,
        "gamma": gamma,
        "gamma_min": gamma_min,
        "gamma_max": gamma_max,
        "out_type": out_type,
        "variance_epsilon": variance_epsilon,
        "scale_after_normalization": scale_after_normalization,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QuantizedBatchNormWithGlobalNormalization"] = quantized_batch_norm_with_global_normalization_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.QuantizedBatchNormWithGlobalNormalization' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedBatchNormWithGlobalNormalization'.")

check_valid('tf.raw_ops.QuantizedBatchNormWithGlobalNormalization', generated_inputs['tf.raw_ops.QuantizedBatchNormWithGlobalNormalization'], lib="tf", suffix=0)
