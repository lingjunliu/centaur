
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_quantizedinstancenorm_inputs():
    list_of_inputs = []

    # Input 1, valid qint8
    x = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.int8)
    x_min = np.array(-10.0, dtype=np.float32)
    x_max = np.array(10.0, dtype=np.float32)
    output_range_given = False
    given_y_min = 0.0
    given_y_max = 0.0
    variance_epsilon = 1e-05
    min_separation = 0.001
    
    qx = tf.quantization.quantize(x, x_min, x_max, tf.dtypes.qint8)
    x_quantized = qx[0]
    
    input_dict = {
        "x": x_quantized.numpy(),
        "x_min": x_min,
        "x_max": x_max,
        "output_range_given": output_range_given,
        "given_y_min": given_y_min,
        "given_y_max": given_y_max,
        "variance_epsilon": variance_epsilon,
        "min_separation": min_separation,
        "name": name,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid, quint8
    x = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.uint8)
    x_min = np.array(0.0, dtype=np.float32)
    x_max = np.array(255.0, dtype=np.float32)
    output_range_given = True
    given_y_min = -1.0
    given_y_max = 1.0
    variance_epsilon = 1e-04
    min_separation = 0.002
    name = "instance_norm_2"

    qx = tf.quantization.quantize(x, x_min, x_max, tf.dtypes.quint8)
    x_quantized = qx[0]

    input_dict = {
        "x": x_quantized.numpy(),
        "x_min": x_min,
        "x_max": x_max,
        "output_range_given": output_range_given,
        "given_y_min": given_y_min,
        "given_y_max": given_y_max,
        "variance_epsilon": variance_epsilon,
        "min_separation": min_separation,
        "name": name,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid, qint32
    x = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.int32)
    x_min = np.array(-100.0, dtype=np.float32)
    x_max = np.array(100.0, dtype=np.float32)
    output_range_given = False
    given_y_min = 0.0
    given_y_max = 0.0
    variance_epsilon = 1e-06
    min_separation = 0.0005
    name = "instance_norm_3"
    
    qx = tf.quantization.quantize(x, x_min, x_max, tf.dtypes.qint32)
    x_quantized = qx[0]

    input_dict = {
        "x": x_quantized.numpy(),
        "x_min": x_min,
        "x_max": x_max,
        "output_range_given": output_range_given,
        "given_y_min": given_y_min,
        "given_y_max": given_y_max,
        "variance_epsilon": variance_epsilon,
        "min_separation": min_separation,
        "name": name,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid, qint16
    x = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.int16)
    x_min = np.array(-50.0, dtype=np.float32)
    x_max = np.array(50.0, dtype=np.float32)
    output_range_given = True
    given_y_min = -0.5
    given_y_max = 0.5
    variance_epsilon = 1e-07
    min_separation = 0.0001
    name = "instance_norm_4"
    
    qx = tf.quantization.quantize(x, x_min, x_max, tf.dtypes.qint16)
    x_quantized = qx[0]


    input_dict = {
        "x": x_quantized.numpy(),
        "x_min": x_min,
        "x_max": x_max,
        "output_range_given": output_range_given,
        "given_y_min": given_y_min,
        "given_y_max": given_y_max,
        "variance_epsilon": variance_epsilon,
        "min_separation": min_separation,
        "name": name,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid, quint16
    x = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.uint16)
    x_min = np.array(0.0, dtype=np.float32)
    x_max = np.array(65535.0, dtype=np.float32)
    output_range_given = False
    given_y_min = 0.0
    given_y_max = 0.0
    variance_epsilon = 1e-08
    min_separation = 0.00005
    name = "instance_norm_5"

    qx = tf.quantization.quantize(x, x_min, x_max, tf.dtypes.quint16)
    x_quantized = qx[0]
    
    input_dict = {
        "x": x_quantized.numpy(),
        "x_min": x_min,
        "x_max": x_max,
        "output_range_given": output_range_given,
        "given_y_min": given_y_min,
        "given_y_max": given_y_max,
        "variance_epsilon": variance_epsilon,
        "min_separation": min_separation,
        "name": name,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid, qint8, different shape
    x = np.array([[[[1, 2, 3], [4, 5, 6]]]], dtype=np.int8)
    x_min = np.array(-10.0, dtype=np.float32)
    x_max = np.array(10.0, dtype=np.float32)
    output_range_given = False
    given_y_min = 0.0
    given_y_max = 0.0
    variance_epsilon = 1e-05
    min_separation = 0.001
    name = None

    qx = tf.quantization.quantize(x, x_min, x_max, tf.dtypes.qint8)
    x_quantized = qx[0]

    input_dict = {
        "x": x_quantized.numpy(),
        "x_min": x_min,
        "x_max": x_max,
        "output_range_given": output_range_given,
        "given_y_min": given_y_min,
        "given_y_max": given_y_max,
        "variance_epsilon": variance_epsilon,
        "min_separation": min_separation,
        "name": name,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid, quint8, different shape
    x = np.array([[[[1, 2], [3, 4], [5, 6]]]], dtype=np.uint8)
    x_min = np.array(0.0, dtype=np.float32)
    x_max = np.array(200.0, dtype=np.float32)
    output_range_given = True
    given_y_min = -0.75
    given_y_max = 0.75
    variance_epsilon = 1e-04
    min_separation = 0.002
    name = "instance_norm_7"
    
    qx = tf.quantization.quantize(x, x_min, x_max, tf.dtypes.quint8)
    x_quantized = qx[0]


    input_dict = {
        "x": x_quantized.numpy(),
        "x_min": x_min,
        "x_max": x_max,
        "output_range_given": output_range_given,
        "given_y_min": given_y_min,
        "given_y_max": given_y_max,
        "variance_epsilon": variance_epsilon,
        "min_separation": min_separation,
        "name": name,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid, qint32, different shape
    x = np.array([[[[1, 2]], [[3, 4]]]], dtype=np.int32)
    x_min = np.array(-50.0, dtype=np.float32)
    x_max = np.array(50.0, dtype=np.float32)
    output_range_given = False
    given_y_min = 0.0
    given_y_max = 0.0
    variance_epsilon = 1e-06
    min_separation = 0.0005
    name = "instance_norm_8"

    qx = tf.quantization.quantize(x, x_min, x_max, tf.dtypes.qint32)
    x_quantized = qx[0]

    input_dict = {
        "x": x_quantized.numpy(),
        "x_min": x_min,
        "x_max": x_max,
        "output_range_given": output_range_given,
        "given_y_min": given_y_min,
        "given_y_max": given_y_max,
        "variance_epsilon": variance_epsilon,
        "min_separation": min_separation,
        "name": name,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid, qint16, different shape
    x = np.array([[[[1, 2, 3, 4]]]], dtype=np.int16)
    x_min = np.array(-25.0, dtype=np.float32)
    x_max = np.array(25.0, dtype=np.float32)
    output_range_given = True
    given_y_min = -0.25
    given_y_max = 0.25
    variance_epsilon = 1e-07
    min_separation = 0.0001
    name = "instance_norm_9"
    
    qx = tf.quantization.quantize(x, x_min, x_max, tf.dtypes.qint16)
    x_quantized = qx[0]


    input_dict = {
        "x": x_quantized.numpy(),
        "x_min": x_min,
        "x_max": x_max,
        "output_range_given": output_range_given,
        "given_y_min": given_y_min,
        "given_y_max": given_y_max,
        "variance_epsilon": variance_epsilon,
        "min_separation": min_separation,
        "name": name,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid, quint16, different shape
    x = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.uint16)
    x_min = np.array(0.0, dtype=np.float32)
    x_max = np.array(32767.0, dtype=np.float32)
    output_range_given = False
    given_y_min = 0.0
    given_y_max = 0.0
    variance_epsilon = 1e-08
    min_separation = 0.00005
    name = "instance_norm_10"

    qx = tf.quantization.quantize(x, x_min, x_max, tf.dtypes.quint16)
    x_quantized = qx[0]

    input_dict = {
        "x": x_quantized.numpy(),
        "x_min": x_min,
        "x_max": x_max,
        "output_range_given": output_range_given,
        "given_y_min": given_y_min,
        "given_y_max": given_y_max,
        "variance_epsilon": variance_epsilon,
        "min_separation": min_separation,
        "name": name,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QuantizedInstanceNorm"] = tf_raw_ops_quantizedinstancenorm_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.QuantizedInstanceNorm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedInstanceNorm'.")

check_valid('tf.raw_ops.QuantizedInstanceNorm', generated_inputs['tf.raw_ops.QuantizedInstanceNorm'], lib="tf", suffix=0)
