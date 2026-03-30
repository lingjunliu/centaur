
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np

def tf_raw_ops_quantized_bias_add_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8)
    bias_tensor = np.array([1, 2, 3], dtype=np.int8)
    min_input = np.float32(0.0)
    max_input = np.float32(10.0)
    min_bias = np.float32(0.0)
    max_bias = np.float32(5.0)
    out_type = tf.qint8

    q_input = tf.quantization.quantize(input_tensor, min_input, max_input, tf.qint8).output
    q_bias = tf.quantization.quantize(bias_tensor, min_bias, max_bias, tf.qint8).output

    input_dict = {
        "input": q_input,
        "bias": q_bias,
        "min_input": min_input,
        "max_input": max_input,
        "min_bias": min_bias,
        "max_bias": max_bias,
        "out_type": out_type,
        "name": "quantized_bias_add_1"
    }
    list_of_inputs.append(input_dict)

    # Input 2
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.uint8)
    bias_tensor = np.array([1, 2], dtype=np.uint8)
    min_input = np.float32(0.0)
    max_input = np.float32(255.0)
    min_bias = np.float32(0.0)
    max_bias = np.float32(10.0)
    out_type = tf.quint8

    q_input = tf.quantization.quantize(input_tensor, min_input, max_input, tf.quint8).output
    q_bias = tf.quantization.quantize(bias_tensor, min_bias, max_bias, tf.quint8).output


    input_dict = {
        "input": q_input,
        "bias": q_bias,
        "min_input": min_input,
        "max_input": max_input,
        "min_bias": min_bias,
        "max_bias": max_bias,
        "out_type": out_type,
        "name": "quantized_bias_add_2"
    }
    list_of_inputs.append(input_dict)

    # Input 3
    input_tensor = np.array([[1000, 2000], [3000, 4000]], dtype=np.int32)
    bias_tensor = np.array([100, 200], dtype=np.int32)
    min_input = np.float32(-5000.0)
    max_input = np.float32(5000.0)
    min_bias = np.float32(-1000.0)
    max_bias = np.float32(1000.0)
    out_type = tf.qint32

    q_input = tf.quantization.quantize(input_tensor, min_input, max_input, tf.qint32).output
    q_bias = tf.quantization.quantize(bias_tensor, min_bias, max_bias, tf.qint32).output

    input_dict = {
        "input": q_input,
        "bias": q_bias,
        "min_input": min_input,
        "max_input": max_input,
        "min_bias": min_bias,
        "max_bias": max_bias,
        "out_type": out_type,
        "name": "quantized_bias_add_3"
    }
    list_of_inputs.append(input_dict)

    # Input 4
    input_tensor = np.array([[-10, -20], [-30, -40]], dtype=np.int16)
    bias_tensor = np.array([-1, -2], dtype=np.int16)
    min_input = np.float32(-50.0)
    max_input = np.float32(0.0)
    min_bias = np.float32(-5.0)
    max_bias = np.float32(0.0)
    out_type = tf.qint16

    q_input = tf.quantization.quantize(input_tensor, min_input, max_input, tf.qint16).output
    q_bias = tf.quantization.quantize(bias_tensor, min_bias, max_bias, tf.qint16).output

    input_dict = {
        "input": q_input,
        "bias": q_bias,
        "min_input": min_input,
        "max_input": max_input,
        "min_bias": min_bias,
        "max_bias": max_bias,
        "out_type": out_type,
        "name": "quantized_bias_add_4"
    }
    list_of_inputs.append(input_dict)

    # Input 5
    input_tensor = np.array([[10, 20], [30, 40]], dtype=np.uint16)
    bias_tensor = np.array([1, 2], dtype=np.uint16)
    min_input = np.float32(0.0)
    max_input = np.float32(50.0)
    min_bias = np.float32(0.0)
    max_bias = np.float32(5.0)
    out_type = tf.quint16

    q_input = tf.quantization.quantize(input_tensor, min_input, max_input, tf.quint16).output
    q_bias = tf.quantization.quantize(bias_tensor, min_bias, max_bias, tf.quint16).output

    input_dict = {
        "input": q_input,
        "bias": q_bias,
        "min_input": min_input,
        "max_input": max_input,
        "min_bias": min_bias,
        "max_bias": max_bias,
        "out_type": out_type,
        "name": "quantized_bias_add_5"
    }
    list_of_inputs.append(input_dict)

    # Input 6
    input_tensor = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.int8)
    bias_tensor = np.array([1, 2, 3], dtype=np.int8)
    min_input = np.float32(-10.0)
    max_input = np.float32(10.0)
    min_bias = np.float32(-5.0)
    max_bias = np.float32(5.0)
    out_type = tf.qint8

    q_input = tf.quantization.quantize(input_tensor, min_input, max_input, tf.qint8).output
    q_bias = tf.quantization.quantize(bias_tensor, min_bias, max_bias, tf.qint8).output

    input_dict = {
        "input": q_input,
        "bias": q_bias,
        "min_input": min_input,
        "max_input": max_input,
        "min_bias": min_bias,
        "max_bias": max_bias,
        "out_type": out_type,
        "name": "quantized_bias_add_6"
    }
    list_of_inputs.append(input_dict)

    # Input 7
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    bias_tensor = np.array([5, 6], dtype=np.uint8)
    min_input = np.float32(10.0)
    max_input = np.float32(20.0)
    min_bias = np.float32(5.0)
    max_bias = np.float32(15.0)
    out_type = tf.quint8

    q_input = tf.quantization.quantize(input_tensor, min_input, max_input, tf.quint8).output
    q_bias = tf.quantization.quantize(bias_tensor, min_bias, max_bias, tf.quint8).output

    input_dict = {
        "input": q_input,
        "bias": q_bias,
        "min_input": min_input,
        "max_input": max_input,
        "min_bias": min_bias,
        "max_bias": max_bias,
        "out_type": out_type,
        "name": "quantized_bias_add_7"
    }
    list_of_inputs.append(input_dict)

    # Input 8
    input_tensor = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int8)
    bias_tensor = np.array([1, 2, 3, 4], dtype=np.int8)
    min_input = np.float32(-1.0)
    max_input = np.float32(1.0)
    min_bias = np.float32(-0.5)
    max_bias = np.float32(0.5)
    out_type = tf.qint8

    q_input = tf.quantization.quantize(input_tensor, min_input, max_input, tf.qint8).output
    q_bias = tf.quantization.quantize(bias_tensor, min_bias, max_bias, tf.qint8).output

    input_dict = {
        "input": q_input,
        "bias": q_bias,
        "min_input": min_input,
        "max_input": max_input,
        "min_bias": min_bias,
        "max_bias": max_bias,
        "out_type": out_type,
        "name": "quantized_bias_add_8"
    }
    list_of_inputs.append(input_dict)

    # Input 9
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    bias_tensor = np.array([1, 2], dtype=np.int32)
    min_input = np.float32(-100.0)
    max_input = np.float32(100.0)
    min_bias = np.float32(-50.0)
    max_bias = np.float32(50.0)
    out_type = tf.qint32

    q_input = tf.quantization.quantize(input_tensor, min_input, max_input, tf.qint32).output
    q_bias = tf.quantization.quantize(bias_tensor, min_bias, max_bias, tf.qint32).output

    input_dict = {
        "input": q_input,
        "bias": q_bias,
        "min_input": min_input,
        "max_input": max_input,
        "min_bias": min_bias,
        "max_bias": max_bias,
        "out_type": out_type,
        "name": "quantized_bias_add_9"
    }
    list_of_inputs.append(input_dict)

    # Input 10
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    bias_tensor = np.array([5, 6], dtype=np.uint16)
    min_input = np.float32(0.0)
    max_input = np.float32(100.0)
    min_bias = np.float32(0.0)
    max_bias = np.float32(50.0)
    out_type = tf.quint16

    q_input = tf.quantization.quantize(input_tensor, min_input, max_input, tf.quint16).output
    q_bias = tf.quantization.quantize(bias_tensor, min_bias, max_bias, tf.quint16).output

    input_dict = {
        "input": q_input,
        "bias": q_bias,
        "min_input": min_input,
        "max_input": max_input,
        "min_bias": min_bias,
        "max_bias": max_bias,
        "out_type": out_type,
        "name": "quantized_bias_add_10"
    }
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QuantizedBiasAdd"] = tf_raw_ops_quantized_bias_add_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QuantizedBiasAdd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedBiasAdd'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.QuantizedBiasAdd', generated_inputs['tf.raw_ops.QuantizedBiasAdd'], lib="tf", suffix=0)
