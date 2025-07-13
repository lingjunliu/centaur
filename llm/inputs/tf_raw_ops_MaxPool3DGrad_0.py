
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_MaxPool3DGrad_inputs():
    list_of_inputs = []

    # Input 1
    orig_input = np.random.rand(1, 3, 3, 3, 1).astype(np.float32)
    orig_output = np.random.rand(1, 2, 2, 2, 1).astype(np.float32)
    grad = np.random.rand(1, 2, 2, 2, 1).astype(np.float32)
    ksize = [1, 2, 2, 2, 1]
    strides = [1, 1, 1, 1, 1]
    padding = "VALID"
    data_format = "NDHWC"
    name = None

    input_dict = {
        "orig_input": tf.convert_to_tensor(orig_input, dtype=tf.float32).numpy(),
        "orig_output": tf.convert_to_tensor(orig_output, dtype=tf.float32).numpy(),
        "grad": tf.convert_to_tensor(grad, dtype=tf.float32).numpy(),
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    orig_input = np.random.rand(1, 5, 5, 5, 3).astype(np.float32)
    orig_output = np.random.rand(1, 3, 3, 3, 3).astype(np.float32)
    grad = np.random.rand(1, 3, 3, 3, 3).astype(np.float32)
    ksize = [1, 3, 3, 3, 1]
    strides = [1, 2, 2, 2, 1]
    padding = "VALID"
    data_format = "NDHWC"
    name = "maxpool3d_grad"

    input_dict = {
        "orig_input": tf.convert_to_tensor(orig_input, dtype=tf.float32).numpy(),
        "orig_output": tf.convert_to_tensor(orig_output, dtype=tf.float32).numpy(),
        "grad": tf.convert_to_tensor(grad, dtype=tf.float32).numpy(),
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    orig_input = np.random.rand(2, 7, 7, 7, 1).astype(np.float32)
    orig_output = np.random.rand(2, 7, 7, 7, 1).astype(np.float32)
    grad = np.random.rand(2, 7, 7, 7, 1).astype(np.float32)
    ksize = [1, 1, 1, 1, 1]
    strides = [1, 1, 1, 1, 1]
    padding = "SAME"
    data_format = "NDHWC"
    name = None

    input_dict = {
        "orig_input": tf.convert_to_tensor(orig_input, dtype=tf.float32).numpy(),
        "orig_output": tf.convert_to_tensor(orig_output, dtype=tf.float32).numpy(),
        "grad": tf.convert_to_tensor(grad, dtype=tf.float32).numpy(),
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    orig_input = np.random.rand(1, 4, 4, 4, 2).astype(np.float32)
    orig_output = np.random.rand(1, 2, 2, 2, 2).astype(np.float32)
    grad = np.random.rand(1, 2, 2, 2, 2).astype(np.float32)
    ksize = [1, 3, 3, 3, 1]
    strides = [1, 2, 2, 2, 1]
    padding = "VALID"
    data_format = "NDHWC"
    name = None

    input_dict = {
        "orig_input": tf.convert_to_tensor(orig_input, dtype=tf.float32).numpy(),
        "orig_output": tf.convert_to_tensor(orig_output, dtype=tf.float32).numpy(),
        "grad": tf.convert_to_tensor(grad, dtype=tf.float32).numpy(),
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    orig_input = np.random.rand(1, 3, 3, 3, 1).astype(np.float32)
    orig_output = np.random.rand(1, 3, 3, 3, 1).astype(np.float32)
    grad = np.random.rand(1, 3, 3, 3, 1).astype(np.float32)
    ksize = [1, 1, 1, 1, 1]
    strides = [1, 1, 1, 1, 1]
    padding = "SAME"
    data_format = "NDHWC"
    name = None

    input_dict = {
        "orig_input": tf.convert_to_tensor(orig_input, dtype=tf.float32).numpy(),
        "orig_output": tf.convert_to_tensor(orig_output, dtype=tf.float32).numpy(),
        "grad": tf.convert_to_tensor(grad, dtype=tf.float32).numpy(),
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different data format NCDHW
    orig_input = np.random.rand(1, 3, 5, 5, 5).astype(np.float32)
    orig_output = np.random.rand(1, 3, 3, 3, 3).astype(np.float32)
    grad = np.random.rand(1, 3, 3, 3, 3).astype(np.float32)
    ksize = [1, 1, 3, 3, 3]
    strides = [1, 1, 2, 2, 2]
    padding = "VALID"
    data_format = "NCDHW"
    name = None

    input_dict = {
        "orig_input": tf.convert_to_tensor(orig_input, dtype=tf.float32).numpy(),
        "orig_output": tf.convert_to_tensor(orig_output, dtype=tf.float32).numpy(),
        "grad": tf.convert_to_tensor(grad, dtype=tf.float32).numpy(),
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7: Half type
    orig_input = np.random.rand(1, 3, 3, 3, 1).astype(np.float16)
    orig_output = np.random.rand(1, 2, 2, 2, 1).astype(np.float16)
    grad = np.random.rand(1, 2, 2, 2, 1).astype(np.float16)
    ksize = [1, 2, 2, 2, 1]
    strides = [1, 1, 1, 1, 1]
    padding = "VALID"
    data_format = "NDHWC"
    name = None

    input_dict = {
        "orig_input": tf.convert_to_tensor(orig_input, dtype=tf.float16).numpy(),
        "orig_output": tf.convert_to_tensor(orig_output, dtype=tf.float16).numpy(),
        "grad": tf.convert_to_tensor(grad, dtype=tf.float16).numpy(),
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float16 type
    orig_input = np.random.rand(1, 3, 3, 3, 1).astype(np.float16)
    orig_output = np.random.rand(1, 2, 2, 2, 1).astype(np.float16)
    grad = np.random.rand(1, 2, 2, 2, 1).astype(np.float16)
    ksize = [1, 2, 2, 2, 1]
    strides = [1, 1, 1, 1, 1]
    padding = "VALID"
    data_format = "NDHWC"
    name = None

    input_dict = {
        "orig_input": tf.convert_to_tensor(orig_input, dtype=tf.float16).numpy(),
        "orig_output": tf.convert_to_tensor(orig_output, dtype=tf.float16).numpy(),
        "grad": tf.convert_to_tensor(grad, dtype=tf.float16).numpy(),
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger batch size
    orig_input = np.random.rand(4, 3, 3, 3, 1).astype(np.float32)
    orig_output = np.random.rand(4, 2, 2, 2, 1).astype(np.float32)
    grad = np.random.rand(4, 2, 2, 2, 1).astype(np.float32)
    ksize = [1, 2, 2, 2, 1]
    strides = [1, 1, 1, 1, 1]
    padding = "VALID"
    data_format = "NDHWC"
    name = None

    input_dict = {
        "orig_input": tf.convert_to_tensor(orig_input, dtype=tf.float32).numpy(),
        "orig_output": tf.convert_to_tensor(orig_output, dtype=tf.float32).numpy(),
        "grad": tf.convert_to_tensor(grad, dtype=tf.float32).numpy(),
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different ksize and strides
    orig_input = np.random.rand(1, 8, 8, 8, 1).astype(np.float32)
    orig_output = np.random.rand(1, 4, 4, 4, 1).astype(np.float32)
    grad = np.random.rand(1, 4, 4, 4, 1).astype(np.float32)
    ksize = [1, 3, 3, 3, 1]
    strides = [1, 2, 2, 2, 1]
    padding = "VALID"
    data_format = "NDHWC"
    name = None

    input_dict = {
        "orig_input": tf.convert_to_tensor(orig_input, dtype=tf.float32).numpy(),
        "orig_output": tf.convert_to_tensor(orig_output, dtype=tf.float32).numpy(),
        "grad": tf.convert_to_tensor(grad, dtype=tf.float32).numpy(),
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.MaxPool3DGrad"] = tf_raw_ops_MaxPool3DGrad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MaxPool3DGrad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MaxPool3DGrad'.")

check_valid('tf.raw_ops.MaxPool3DGrad', generated_inputs['tf.raw_ops.MaxPool3DGrad'], lib="tf", suffix=0)
