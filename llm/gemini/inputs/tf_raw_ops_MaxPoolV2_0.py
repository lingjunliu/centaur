
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def tf_raw_ops_MaxPoolV2_inputs():
    list_of_inputs = []

    # Input 1: Float32, NHWC, VALID
    list_of_inputs.append({
        'data_format': 'NHWC',
        'name': 'pool_1',
        'input': np.random.randn(2, 8, 8, 3).astype(np.float32),
        'ksize': np.array([1, 2, 2, 1], dtype=np.int32),
        'strides': np.array([1, 2, 2, 1], dtype=np.int32),
        'padding': 'VALID'
    })

    # Input 2: Float64, NHWC, SAME
    list_of_inputs.append({
        'data_format': 'NHWC',
        'name': 'pool_2',
        'input': np.random.randn(1, 10, 10, 1).astype(np.float64),
        'ksize': np.array([1, 3, 3, 1], dtype=np.int32),
        'strides': np.array([1, 1, 1, 1], dtype=np.int32),
        'padding': 'SAME'
    })

    # Input 3: Int8, NHWC, VALID
    list_of_inputs.append({
        'data_format': 'NHWC',
        'name': 'pool_3',
        'input': np.random.randint(-128, 127, size=(4, 8, 8, 3)).astype(np.int8),
        'ksize': np.array([1, 2, 2, 1], dtype=np.int32),
        'strides': np.array([1, 2, 2, 1], dtype=np.int32),
        'padding': 'VALID'
    })

    # Input 4: UInt8, NHWC, SAME
    list_of_inputs.append({
        'data_format': 'NHWC',
        'name': 'pool_4',
        'input': np.random.randint(0, 255, size=(1, 5, 5, 2)).astype(np.uint8),
        'ksize': np.array([1, 2, 2, 1], dtype=np.int32),
        'strides': np.array([1, 1, 1, 1], dtype=np.int32),
        'padding': 'SAME'
    })

    # Input 5: Int16, NHWC, VALID
    list_of_inputs.append({
        'data_format': 'NHWC',
        'name': 'pool_5',
        'input': np.random.randint(-32768, 32767, size=(2, 6, 6, 4)).astype(np.int16),
        'ksize': np.array([1, 3, 3, 1], dtype=np.int32),
        'strides': np.array([1, 2, 2, 1], dtype=np.int32),
        'padding': 'VALID'
    })

    # Input 6: Float16, NHWC, SAME
    list_of_inputs.append({
        'data_format': 'NHWC',
        'name': 'pool_6',
        'input': np.random.randn(1, 4, 4, 1).astype(np.float16),
        'ksize': np.array([1, 2, 2, 1], dtype=np.int32),
        'strides': np.array([1, 1, 1, 1], dtype=np.int32),
        'padding': 'SAME'
    })

    # Input 7: Float32, NHWC, VALID
    list_of_inputs.append({
        'data_format': 'NHWC',
        'name': 'pool_7',
        'input': np.random.randn(2, 8, 8, 2).astype(np.float32),
        'ksize': np.array([1, 2, 2, 1], dtype=np.int32),
        'strides': np.array([1, 1, 1, 1], dtype=np.int32),
        'padding': 'VALID'
    })

    # Input 8: Int64, NHWC, SAME
    list_of_inputs.append({
        'data_format': 'NHWC',
        'name': 'pool_8',
        'input': np.random.randint(-1000, 1000, size=(1, 12, 12, 2)).astype(np.int64),
        'ksize': np.array([1, 3, 3, 1], dtype=np.int32),
        'strides': np.array([1, 2, 2, 1], dtype=np.int32),
        'padding': 'SAME'
    })

    # Input 9: Float32, NHWC, VALID
    list_of_inputs.append({
        'data_format': 'NHWC',
        'name': 'pool_9',
        'input': np.random.randn(3, 16, 16, 3).astype(np.float32),
        'ksize': np.array([1, 4, 4, 1], dtype=np.int32),
        'strides': np.array([1, 4, 4, 1], dtype=np.int32),
        'padding': 'VALID'
    })

    # Input 10: Int32, NHWC, SAME
    list_of_inputs.append({
        'data_format': 'NHWC',
        'name': 'pool_10',
        'input': np.random.randint(-50, 50, size=(1, 14, 14, 4)).astype(np.int32),
        'ksize': np.array([1, 3, 3, 1], dtype=np.int32),
        'strides': np.array([1, 2, 2, 1], dtype=np.int32),
        'padding': 'SAME'
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.MaxPoolV2"] = tf_raw_ops_MaxPoolV2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.MaxPoolV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MaxPoolV2'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.MaxPoolV2', generated_inputs['tf.raw_ops.MaxPoolV2'], lib="tf", suffix=0)
