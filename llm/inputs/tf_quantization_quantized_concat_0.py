
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_quantized_concat_inputs():
    list_of_inputs = []

    def convert_to_tensor_list(numpy_arrays, dtype=tf.int8):
        if not isinstance(numpy_arrays, list):
            numpy_arrays = [numpy_arrays]
        return [tf.convert_to_tensor(arr, dtype=dtype) for arr in numpy_arrays]

    def convert_to_float_tensor_list(numpy_arrays):
        if not isinstance(numpy_arrays, list):
            numpy_arrays = [numpy_arrays]
        return [tf.convert_to_tensor(np.array(arr).reshape(1), dtype=tf.float32) for arr in numpy_arrays]

    # Input 1
    concat_dim = np.int32(0)
    values = [np.array([[1, 2], [3, 4]], dtype=np.int8), np.array([[5, 6], [7, 8]], dtype=np.int8)]
    input_mins = [np.float32(0.0), np.float32(0.0)]
    input_maxes = [np.float32(5.0), np.float32(8.0)]
    input_dict = {"concat_dim": concat_dim, "values": convert_to_tensor_list(values), "input_mins": convert_to_float_tensor_list(input_mins), "input_maxes": convert_to_float_tensor_list(input_maxes), "name": "concat1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    concat_dim = np.int32(1)
    values = [np.array([[1, 2], [3, 4]], dtype=np.int8), np.array([[5, 6], [7, 8]], dtype=np.int8)]
    input_mins = [np.float32(-1.0), np.float32(-5.0)]
    input_maxes = [np.float32(5.0), np.float32(8.0)]
    input_dict = {"concat_dim": concat_dim, "values": convert_to_tensor_list(values), "input_mins": convert_to_float_tensor_list(input_mins), "input_maxes": convert_to_float_tensor_list(input_maxes), "name": "concat2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    concat_dim = np.int32(0)
    values = [np.array([1, 2, 3], dtype=np.int8), np.array([4, 5, 6], dtype=np.int8)]
    input_mins = [np.float32(0.0), np.float32(0.0)]
    input_maxes = [np.float32(6.0), np.float32(6.0)]
    input_dict = {"concat_dim": concat_dim, "values": convert_to_tensor_list(values), "input_mins": convert_to_float_tensor_list(input_mins), "input_maxes": convert_to_float_tensor_list(input_maxes), "name": "concat3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    concat_dim = np.int32(0)
    values = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int8), np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.int8)]
    input_mins = [np.float32(0.0), np.float32(0.0)]
    input_maxes = [np.float32(8.0), np.float32(16.0)]
    input_dict = {"concat_dim": concat_dim, "values": convert_to_tensor_list(values), "input_mins": convert_to_float_tensor_list(input_mins), "input_maxes": convert_to_float_tensor_list(input_maxes), "name": "concat4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    concat_dim = np.int32(1)
    values = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int8), np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.int8)]
    input_mins = [np.float32(-1.0), np.float32(-9.0)]
    input_maxes = [np.float32(8.0), np.float32(16.0)]
    input_dict = {"concat_dim": concat_dim, "values": convert_to_tensor_list(values), "input_mins": convert_to_float_tensor_list(input_mins), "input_maxes": convert_to_float_tensor_list(input_maxes), "name": "concat5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    concat_dim = np.int32(2)
    values = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int8), np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.int8)]
    input_mins = [np.float32(1.0), np.float32(9.0)]
    input_maxes = [np.float32(8.0), np.float32(16.0)]
    input_dict = {"concat_dim": concat_dim, "values": convert_to_tensor_list(values), "input_mins": convert_to_float_tensor_list(input_mins), "input_maxes": convert_to_float_tensor_list(input_maxes), "name": "concat6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Three tensors
    concat_dim = np.int32(0)
    values = [np.array([[1, 2]], dtype=np.int8), np.array([[3, 4]], dtype=np.int8), np.array([[5, 6]], dtype=np.int8)]
    input_mins = [np.float32(0.0), np.float32(0.0), np.float32(0.0)]
    input_maxes = [np.float32(2.0), np.float32(4.0), np.float32(6.0)]
    input_dict = {"concat_dim": concat_dim, "values": convert_to_tensor_list(values), "input_mins": convert_to_float_tensor_list(input_mins), "input_maxes": convert_to_float_tensor_list(input_maxes), "name": "concat7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: negative values
    concat_dim = np.int32(1)
    values = [np.array([[-1, 2], [3, -4]], dtype=np.int8), np.array([[5, -6], [-7, 8]], dtype=np.int8)]
    input_mins = [np.float32(-4.0), np.float32(-7.0)]
    input_maxes = [np.float32(3.0), np.float32(8.0)]
    input_dict = {"concat_dim": concat_dim, "values": convert_to_tensor_list(values), "input_mins": convert_to_float_tensor_list(input_mins), "input_maxes": convert_to_float_tensor_list(input_maxes), "name": "concat8"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Three dimensional input
    concat_dim = np.int32(1)
    values = [np.array([[[1, 2]], [[3, 4]]], dtype=np.int8), np.array([[[5, 6]], [[7, 8]]], dtype=np.int8)]
    input_mins = [np.float32(0.0), np.float32(0.0)]
    input_maxes = [np.float32(4.0), np.float32(8.0)]
    input_dict = {"concat_dim": concat_dim, "values": convert_to_tensor_list(values), "input_mins": convert_to_float_tensor_list(input_mins), "input_maxes": convert_to_float_tensor_list(input_maxes), "name": "concat9"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10 : Different data type
    concat_dim = np.int32(0)
    values = [np.array([[1, 2], [3, 4]], dtype=np.int16), np.array([[5, 6], [7, 8]], dtype=np.int16)]
    input_mins = [np.float32(0.0), np.float32(0.0)]
    input_maxes = [np.float32(5.0), np.float32(8.0)]
    input_dict = {"concat_dim": concat_dim, "values": convert_to_tensor_list(values, dtype=tf.int16), "input_mins": convert_to_float_tensor_list(input_mins), "input_maxes": convert_to_float_tensor_list(input_maxes), "name": "concat10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.quantization.quantized_concat"] = tf_quantized_concat_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.quantization.quantized_concat' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.quantization.quantized_concat'.")

check_valid('tf.quantization.quantized_concat', generated_inputs['tf.quantization.quantized_concat'], lib="tf", suffix=0)
