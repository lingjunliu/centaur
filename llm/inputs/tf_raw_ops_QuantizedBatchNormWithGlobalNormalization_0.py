
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def tf_raw_ops_quantized_batch_norm_with_global_normalization_inputs():
    """
    Generates a list of valid inputs for the
    tf.raw_ops.QuantizedBatchNormWithGlobalNormalization function.
    This version creates correctly typed quantized tensors to resolve the
    InvalidArgumentError from TensorFlow.
    """
    list_of_inputs = []

    def generate_one_input(tf_input_type, shape, scale_after, epsilon, name, out_type, custom_ranges=None):
        channels = shape[-1]

        ranges = custom_ranges or {
            't': (-1.0, 1.0), 'm': (-0.5, 0.5), 'v': (0.0, 1.0),
            'beta': (-0.1, 0.1), 'gamma': (0.9, 1.1)
        }
        
        # Helper to create a quantized tensor
        def quantize(float_data, min_val, max_val):
            # For quint types, the float data must be non-negative
            if tf_input_type.is_unsigned:
                float_data = np.abs(float_data)
                min_val, max_val = abs(min_val), abs(max_val)

            q, _, _ = tf.quantization.quantize(
                tf.constant(float_data, dtype=tf.float32), 
                min_range=tf.constant(min_val, dtype=tf.float32), 
                max_range=tf.constant(max_val, dtype=tf.float32),
                T=tf_input_type, 
                mode='SCALED'
            )
            return q, min_val, max_val

        t_float = np.random.uniform(-1.0, 1.0, size=shape).astype(np.float32)
        m_float = np.random.uniform(-1.0, 1.0, size=(channels,)).astype(np.float32)
        v_float = np.random.uniform(0.0, 1.0, size=(channels,)).astype(np.float32) # Variance is non-negative
        beta_float = np.random.uniform(-1.0, 1.0, size=(channels,)).astype(np.float32)
        gamma_float = np.random.uniform(-1.0, 1.0, size=(channels,)).astype(np.float32)

        t_q, t_min_val, t_max_val = quantize(t_float, ranges['t'][0], ranges['t'][1])
        m_q, m_min_val, m_max_val = quantize(m_float, ranges['m'][0], ranges['m'][1])
        v_q, v_min_val, v_max_val = quantize(v_float, ranges['v'][0], ranges['v'][1])
        beta_q, beta_min_val, beta_max_val = quantize(beta_float, ranges['beta'][0], ranges['beta'][1])
        gamma_q, gamma_min_val, gamma_max_val = quantize(gamma_float, ranges['gamma'][0], ranges['gamma'][1])

        input_dict = {
            't': t_q,
            't_min': np.array(t_min_val, dtype=np.float32),
            't_max': np.array(t_max_val, dtype=np.float32),
            'm': m_q,
            'm_min': np.array(m_min_val, dtype=np.float32),
            'm_max': np.array(m_max_val, dtype=np.float32),
            'v': v_q,
            'v_min': np.array(v_min_val, dtype=np.float32),
            'v_max': np.array(v_max_val, dtype=np.float32),
            'beta': beta_q,
            'beta_min': np.array(beta_min_val, dtype=np.float32),
            'beta_max': np.array(beta_max_val, dtype=np.float32),
            'gamma': gamma_q,
            'gamma_min': np.array(gamma_min_val, dtype=np.float32),
            'gamma_max': np.array(gamma_max_val, dtype=np.float32),
            'out_type': out_type,
            'variance_epsilon': epsilon,
            'scale_after_normalization': scale_after,
            'name': name
        }
        # The min/max inputs must be tensors according to the signature, so we convert them back
        for key in ['t_min', 't_max', 'm_min', 'm_max', 'v_min', 'v_max', 'beta_min', 'beta_max', 'gamma_min', 'gamma_max']:
            input_dict[key] = tf.constant(input_dict[key])

        return input_dict

    list_of_inputs.append(copy.deepcopy(generate_one_input(tf.qint8, (1, 2, 2, 3), True, 0.001, 'case1_qint8', tf.qint8)))
    list_of_inputs.append(copy.deepcopy(generate_one_input(tf.quint8, (2, 3, 3, 4), False, 1e-5, 'case2_quint8_no_scale', tf.quint8)))
    list_of_inputs.append(copy.deepcopy(generate_one_input(tf.qint32, (1, 1, 1, 5), True, 1e-4, 'case3_qint32', tf.qint32)))
    list_of_inputs.append(copy.deepcopy(generate_one_input(tf.qint16, (1, 4, 4, 2), True, 0.01, 'case4_qint16', tf.qint16)))
    custom_ranges_5 = {'t': (0.0, 100.0), 'm': (10.0, 20.0), 'v': (1.0, 5.0), 'beta': (-1.0, 1.0), 'gamma': (0.8, 1.2)}
    list_of_inputs.append(copy.deepcopy(generate_one_input(tf.quint16, (3, 1, 2, 6), True, 1e-9, 'case5_quint16', tf.quint16, custom_ranges_5)))
    list_of_inputs.append(copy.deepcopy(generate_one_input(tf.qint8, (1, 2, 2, 2), True, 0.002, 'case6_type_promotion', tf.qint32)))
    list_of_inputs.append(copy.deepcopy(generate_one_input(tf.qint8, (1, 1, 1, 1), True, 0.001, 'case7_minimal_shape', tf.qint8)))
    list_of_inputs.append(copy.deepcopy(generate_one_input(tf.quint8, (1, 3, 3, 3), True, 0.005, 'case8_type_conversion', tf.qint8)))
    list_of_inputs.append(copy.deepcopy(generate_one_input(tf.qint16, (1, 2, 2, 10), True, 1e-3, 'case9_large_channels', tf.qint16)))
    custom_ranges_10 = {'t': (-10.0, -5.0), 'm': (-2.0, -1.0), 'v': (0.1, 0.5), 'beta': (-0.5, -0.1), 'gamma': (0.8, 1.2)}
    list_of_inputs.append(copy.deepcopy(generate_one_input(tf.qint8, (1, 3, 3, 4), True, 0.001, 'case10_negative_ranges', tf.qint8, custom_ranges_10)))
    list_of_inputs.append(copy.deepcopy(generate_one_input(tf.qint16, (1, 4, 4, 2), True, 0.01, None, tf.qint16)))

    return list_of_inputs

generated_inputs["tf.raw_ops.QuantizedBatchNormWithGlobalNormalization"] = tf_raw_ops_quantized_batch_norm_with_global_normalization_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QuantizedBatchNormWithGlobalNormalization' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedBatchNormWithGlobalNormalization'.")

check_valid('tf.raw_ops.QuantizedBatchNormWithGlobalNormalization', generated_inputs['tf.raw_ops.QuantizedBatchNormWithGlobalNormalization'], lib="tf", suffix=0)
