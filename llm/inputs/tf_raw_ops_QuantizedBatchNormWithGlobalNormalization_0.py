
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_quantized_batch_norm_with_global_normalization_inputs():
    list_of_inputs = []

    # Input 1
    t = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.int8)
    t_min = np.array(0.0, dtype=np.float32)
    t_max = np.array(10.0, dtype=np.float32)
    m = np.array([1, 6], dtype=np.int8)
    m_min = np.array(1.0, dtype=np.float32)
    m_max = np.array(7.0, dtype=np.float32)
    v = np.array([0, 0], dtype=np.int8)
    v_min = np.array(0.0, dtype=np.float32)
    v_max = np.array(1.0, dtype=np.float32)
    beta = np.array([0, 0], dtype=np.int8)
    beta_min = np.array(0.0, dtype=np.float32)
    beta_max = np.array(0.0, dtype=np.float32)
    gamma = np.array([1, 1], dtype=np.int8)
    gamma_min = np.array(1.0, dtype=np.float32)
    gamma_max = np.array(1.0, dtype=np.float32)
    out_type = tf.qint8
    variance_epsilon = 0.001
    scale_after_normalization = True
    t = tf.constant(t, dtype=tf.qint8)
    m = tf.constant(m, dtype=tf.qint8)
    v = tf.constant(v, dtype=tf.qint8)
    beta = tf.constant(beta, dtype=tf.qint8)
    gamma = tf.constant(gamma, dtype=tf.qint8)

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
        "name": "test_batchnorm"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    t = np.array([[[[10, 20], [30, 40]], [[50, 60], [70, 80]]]], dtype=np.uint8)
    t_min = np.array(0.0, dtype=np.float32)
    t_max = np.array(255.0, dtype=np.float32)
    m = np.array([15, 65], dtype=np.uint8)
    m_min = np.array(10.0, dtype=np.float32)
    m_max = np.array(70.0, dtype=np.float32)
    v = np.array([5, 5], dtype=np.uint8)
    v_min = np.array(0.0, dtype=np.float32)
    v_max = np.array(10.0, dtype=np.float32)
    beta = np.array([5, 5], dtype=np.uint8)
    beta_min = np.array(0.0, dtype=np.float32)
    beta_max = np.array(10.0, dtype=np.float32)
    gamma = np.array([2, 2], dtype=np.uint8)
    gamma_min = np.array(1.0, dtype=np.float32)
    gamma_max = np.array(3.0, dtype=np.float32)
    out_type = tf.quint8
    variance_epsilon = 0.01
    scale_after_normalization = False

    t = tf.constant(t, dtype=tf.quint8)
    m = tf.constant(m, dtype=tf.quint8)
    v = tf.constant(v, dtype=tf.quint8)
    beta = tf.constant(beta, dtype=tf.quint8)
    gamma = tf.constant(gamma, dtype=tf.quint8)

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
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 3
    t = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.int32)
    t_min = np.array(-10.0, dtype=np.float32)
    t_max = np.array(10.0, dtype=np.float32)
    m = np.array([1, 6], dtype=np.int32)
    m_min = np.array(-5.0, dtype=np.float32)
    m_max = np.array(7.0, dtype=np.float32)
    v = np.array([1, 1], dtype=np.int32)
    v_min = np.array(0.0, dtype=np.float32)
    v_max = np.array(2.0, dtype=np.float32)
    beta = np.array([0, 0], dtype=np.int32)
    beta_min = np.array(-1.0, dtype=np.float32)
    beta_max = np.array(1.0, dtype=np.float32)
    gamma = np.array([1, 1], dtype=np.int32)
    gamma_min = np.array(0.5, dtype=np.float32)
    gamma_max = np.array(1.5, dtype=np.float32)
    out_type = tf.qint32
    variance_epsilon = 0.0001
    scale_after_normalization = True

    t = tf.constant(t, dtype=tf.qint32)
    m = tf.constant(m, dtype=tf.qint32)
    v = tf.constant(v, dtype=tf.qint32)
    beta = tf.constant(beta, dtype=tf.qint32)
    gamma = tf.constant(gamma, dtype=tf.qint32)

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
        "name": "test_batchnorm_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    t = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.int16)
    t_min = np.array(-5.0, dtype=np.float32)
    t_max = np.array(15.0, dtype=np.float32)
    m = np.array([2, 5], dtype=np.int16)
    m_min = np.array(-1.0, dtype=np.float32)
    m_max = np.array(6.0, dtype=np.float32)
    v = np.array([1, 1], dtype=np.int16)
    v_min = np.array(0.0, dtype=np.float32)
    v_max = np.array(3.0, dtype=np.float32)
    beta = np.array([-1, 1], dtype=np.int16)
    beta_min = np.array(-2.0, dtype=np.float32)
    beta_max = np.array(2.0, dtype=np.float32)
    gamma = np.array([1, 1], dtype=np.int16)
    gamma_min = np.array(0.75, dtype=np.float32)
    gamma_max = np.array(1.25, dtype=np.float32)
    out_type = tf.qint16
    variance_epsilon = 0.00001
    scale_after_normalization = False

    t = tf.constant(t, dtype=tf.qint16)
    m = tf.constant(m, dtype=tf.qint16)
    v = tf.constant(v, dtype=tf.qint16)
    beta = tf.constant(beta, dtype=tf.qint16)
    gamma = tf.constant(gamma, dtype=tf.qint16)

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
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    t = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.uint16)
    t_min = np.array(0.0, dtype=np.float32)
    t_max = np.array(20.0, dtype=np.float32)
    m = np.array([3, 7], dtype=np.uint16)
    m_min = np.array(2.0, dtype=np.float32)
    m_max = np.array(8.0, dtype=np.float32)
    v = np.array([2, 2], dtype=np.uint16)
    v_min = np.array(0.0, dtype=np.float32)
    v_max = np.array(4.0, dtype=np.float32)
    beta = np.array([0, 0], dtype=np.uint16)
    beta_min = np.array(0.0, dtype=np.float32)
    beta_max = np.array(0.0, dtype=np.float32)
    gamma = np.array([1, 1], dtype=np.uint16)
    gamma_min = np.array(0.9, dtype=np.float32)
    gamma_max = np.array(1.1, dtype=np.float32)
    out_type = tf.quint16
    variance_epsilon = 0.000001
    scale_after_normalization = True

    t = tf.constant(t, dtype=tf.quint16)
    m = tf.constant(m, dtype=tf.quint16)
    v = tf.constant(v, dtype=tf.quint16)
    beta = tf.constant(beta, dtype=tf.quint16)
    gamma = tf.constant(gamma, dtype=tf.quint16)

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
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QuantizedBatchNormWithGlobalNormalization"] = tf_raw_ops_quantized_batch_norm_with_global_normalization_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.QuantizedBatchNormWithGlobalNormalization' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedBatchNormWithGlobalNormalization'.")

check_valid('tf.raw_ops.QuantizedBatchNormWithGlobalNormalization', generated_inputs['tf.raw_ops.QuantizedBatchNormWithGlobalNormalization'], lib="tf", suffix=0)
