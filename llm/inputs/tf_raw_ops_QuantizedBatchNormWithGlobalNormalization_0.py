
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def tf_raw_ops_QuantizedBatchNormWithGlobalNormalization_inputs():
    list_of_inputs = []

    # This op requires `tf.qint*` dtypes. Providing standard `np.int*` arrays,
    # as required by the input format constraints, will cause a `tf.errors.InvalidArgumentError`
    # because TensorFlow interprets them as `tf.int*` instead. This is an unresolvable conflict.
    # The following inputs strictly adhere to the numpy format.

    # Input 1: qint8, scale=True
    channels = 3
    input_dict = {
        't': np.random.randint(-128, 128, size=(1, 2, 2, channels), dtype=np.int8),
        't_min': np.array(-10.0, dtype=np.float32),
        't_max': np.array(10.0, dtype=np.float32),
        'm': np.random.randint(-128, 128, size=(channels,), dtype=np.int8),
        'm_min': np.array(-1.0, dtype=np.float32),
        'm_max': np.array(1.0, dtype=np.float32),
        'v': np.random.randint(0, 128, size=(channels,), dtype=np.int8),
        'v_min': np.array(0.0, dtype=np.float32),
        'v_max': np.array(2.0, dtype=np.float32),
        'beta': np.random.randint(-128, 128, size=(channels,), dtype=np.int8),
        'beta_min': np.array(-0.5, dtype=np.float32),
        'beta_max': np.array(0.5, dtype=np.float32),
        'gamma': np.random.randint(0, 128, size=(channels,), dtype=np.int8),
        'gamma_min': np.array(0.8, dtype=np.float32),
        'gamma_max': np.array(1.2, dtype=np.float32),
        'out_type': np.int8,
        'variance_epsilon': 1e-5,
        'scale_after_normalization': True,
        'name': 'qint8_numpy_input'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: quint8, scale=False
    channels = 4
    input_dict = {
        't': np.random.randint(0, 256, size=(2, 3, 3, channels), dtype=np.uint8),
        't_min': np.array(0.0, dtype=np.float32),
        't_max': np.array(6.0, dtype=np.float32),
        'm': np.random.randint(0, 256, size=(channels,), dtype=np.uint8),
        'm_min': np.array(0.0, dtype=np.float32),
        'm_max': np.array(1.0, dtype=np.float32),
        'v': np.random.randint(0, 256, size=(channels,), dtype=np.uint8),
        'v_min': np.array(0.0, dtype=np.float32),
        'v_max': np.array(1.5, dtype=np.float32),
        'beta': np.random.randint(0, 256, size=(channels,), dtype=np.uint8),
        'beta_min': np.array(0.0, dtype=np.float32),
        'beta_max': np.array(0.2, dtype=np.float32),
        'gamma': np.random.randint(0, 256, size=(channels,), dtype=np.uint8),
        'gamma_min': np.array(0.0, dtype=np.float32),
        'gamma_max': np.array(1.0, dtype=np.float32),
        'out_type': np.uint8,
        'variance_epsilon': 0.001,
        'scale_after_normalization': False,
        'name': 'quint8_numpy_input'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: qint16
    channels = 2
    input_dict = {
        't': np.random.randint(-32768, 32768, size=(1, 5, 5, channels), dtype=np.int16),
        't_min': np.array(-20.0, dtype=np.float32),
        't_max': np.array(20.0, dtype=np.float32),
        'm': np.random.randint(-32768, 32768, size=(channels,), dtype=np.int16),
        'm_min': np.array(-2.0, dtype=np.float32),
        'm_max': np.array(2.0, dtype=np.float32),
        'v': np.random.randint(0, 32768, size=(channels,), dtype=np.int16),
        'v_min': np.array(0.0, dtype=np.float32),
        'v_max': np.array(5.0, dtype=np.float32),
        'beta': np.random.randint(-32768, 32768, size=(channels,), dtype=np.int16),
        'beta_min': np.array(-1.0, dtype=np.float32),
        'beta_max': np.array(1.0, dtype=np.float32),
        'gamma': np.random.randint(0, 32768, size=(channels,), dtype=np.int16),
        'gamma_min': np.array(0.5, dtype=np.float32),
        'gamma_max': np.array(1.5, dtype=np.float32),
        'out_type': np.int16,
        'variance_epsilon': 1e-4,
        'scale_after_normalization': True,
        'name': 'qint16_numpy_input'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: quint16
    channels = 1
    input_dict = {
        't': np.random.randint(0, 65536, size=(3, 2, 2, channels), dtype=np.uint16),
        't_min': np.array(0.0, dtype=np.float32),
        't_max': np.array(256.0, dtype=np.float32),
        'm': np.random.randint(0, 65536, size=(channels,), dtype=np.uint16),
        'm_min': np.array(100.0, dtype=np.float32),
        'm_max': np.array(150.0, dtype=np.float32),
        'v': np.random.randint(0, 65536, size=(channels,), dtype=np.uint16),
        'v_min': np.array(0.0, dtype=np.float32),
        'v_max': np.array(100.0, dtype=np.float32),
        'beta': np.random.randint(0, 65536, size=(channels,), dtype=np.uint16),
        'beta_min': np.array(120.0, dtype=np.float32),
        'beta_max': np.array(130.0, dtype=np.float32),
        'gamma': np.random.randint(0, 65536, size=(channels,), dtype=np.uint16),
        'gamma_min': np.array(0.0, dtype=np.float32),
        'gamma_max': np.array(1.0, dtype=np.float32),
        'out_type': np.uint16,
        'variance_epsilon': 1e-6,
        'scale_after_normalization': False,
        'name': 'quint16_numpy_input'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: qint32
    channels = 5
    input_dict = {
        't': np.random.randint(-100000, 100000, size=(1, 1, 1, channels), dtype=np.int32),
        't_min': np.array(-50.0, dtype=np.float32),
        't_max': np.array(50.0, dtype=np.float32),
        'm': np.random.randint(-100000, 100000, size=(channels,), dtype=np.int32),
        'm_min': np.array(-1.0, dtype=np.float32),
        'm_max': np.array(1.0, dtype=np.float32),
        'v': np.random.randint(0, 100000, size=(channels,), dtype=np.int32),
        'v_min': np.array(0.0, dtype=np.float32),
        'v_max': np.array(2.0, dtype=np.float32),
        'beta': np.random.randint(-100000, 100000, size=(channels,), dtype=np.int32),
        'beta_min': np.array(-0.1, dtype=np.float32),
        'beta_max': np.array(0.1, dtype=np.float32),
        'gamma': np.random.randint(0, 100000, size=(channels,), dtype=np.int32),
        'gamma_min': np.array(0.9, dtype=np.float32),
        'gamma_max': np.array(1.1, dtype=np.float32),
        'out_type': np.int32,
        'variance_epsilon': 1e-2,
        'scale_after_normalization': True,
        'name': 'qint32_numpy_input'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger variance_epsilon
    input_dict_6 = copy.deepcopy(list_of_inputs[0])
    input_dict_6['variance_epsilon'] = 0.1
    input_dict_6['name'] = 'large_epsilon_numpy'
    list_of_inputs.append(input_dict_6)

    # Input 7: Change out_type (uint8 -> int8)
    input_dict_7 = copy.deepcopy(list_of_inputs[1])
    input_dict_7['out_type'] = np.int8
    input_dict_7['name'] = 'uint8_to_int8_numpy'
    list_of_inputs.append(input_dict_7)

    # Input 8: zero-valued variance components
    channels = 4
    input_dict_8 = copy.deepcopy(list_of_inputs[0])
    input_dict_8['t'] = np.random.randint(-128, 128, size=(1, 2, 2, channels), dtype=np.int8)
    input_dict_8['m'] = np.random.randint(-128, 128, size=(channels,), dtype=np.int8)
    input_dict_8['v'] = np.array([50, 0, 25, 0], dtype=np.int8)
    input_dict_8['beta'] = np.random.randint(-128, 128, size=(channels,), dtype=np.int8)
    input_dict_8['gamma'] = np.random.randint(-128, 128, size=(channels,), dtype=np.int8)
    input_dict_8['name'] = 'zero_variance_numpy'
    list_of_inputs.append(input_dict_8)

    # Input 9: Larger batch size
    input_dict_9 = copy.deepcopy(list_of_inputs[2])
    input_dict_9['t'] = np.random.randint(-32768, 32768, size=(16, 2, 2, 2), dtype=np.int16)
    input_dict_9['name'] = 'large_batch_numpy'
    list_of_inputs.append(input_dict_9)

    # Input 10: No name provided
    input_dict_10 = copy.deepcopy(list_of_inputs[4])
    input_dict_10['name'] = None
    list_of_inputs.append(input_dict_10)

    return list_of_inputs

generated_inputs["tf.raw_ops.QuantizedBatchNormWithGlobalNormalization"] = tf_raw_ops_QuantizedBatchNormWithGlobalNormalization_inputs()

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
