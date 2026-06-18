
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_maxpool3dgrad_inputs():
    list_of_inputs = []

    # 1. Float32, NDHWC, VALID, simple 2x2x2
    orig_input = np.arange(8, dtype=np.float32).reshape((1, 2, 2, 2, 1))
    orig_output = np.array([[[[[7.0]]]]], dtype=np.float32)
    grad = np.array([[[[[1.0]]]]], dtype=np.float32)
    ksize = [1, 2, 2, 2, 1]
    strides = [1, 1, 1, 1, 1]
    padding = "VALID"
    data_format = "NDHWC"
    name = "max_pool_3d_grad_1"
    
    list_of_inputs.append({
        'orig_input': orig_input,
        'orig_output': orig_output,
        'grad': grad,
        'ksize': ksize,
        'strides': strides,
        'padding': padding,
        'data_format': data_format,
        'name': name
    })

    # 2. Float32, NDHWC, SAME
    orig_input = np.arange(8, dtype=np.float32).reshape((1, 2, 2, 2, 1))
    orig_output = np.arange(8, dtype=np.float32).reshape((1, 2, 2, 2, 1))
    grad = np.ones((1, 2, 2, 2, 1), dtype=np.float32)
    ksize = [1, 2, 2, 2, 1]
    strides = [1, 1, 1, 1, 1]
    padding = "SAME"
    data_format = "NDHWC"
    name = "max_pool_3d_grad_2"

    list_of_inputs.append({
        'orig_input': orig_input,
        'orig_output': orig_output,
        'grad': grad,
        'ksize': ksize,
        'strides': strides,
        'padding': padding,
        'data_format': data_format,
        'name': name
    })

    # 3. Float32, Negative values, NDHWC, VALID
    orig_input = np.random.uniform(-10, 10, (1, 3, 3, 3, 2)).astype(np.float32)
    orig_output = np.random.uniform(-10, 10, (1, 2, 2, 2, 2)).astype(np.float32)
    grad = np.random.uniform(-1, 1, (1, 2, 2, 2, 2)).astype(np.float32)
    ksize = [1, 2, 2, 2, 1]
    strides = [1, 1, 1, 1, 1]
    padding = "VALID"
    data_format = "NDHWC"
    name = "max_pool_3d_grad_3"

    list_of_inputs.append({
        'orig_input': orig_input,
        'orig_output': orig_output,
        'grad': grad,
        'ksize': ksize,
        'strides': strides,
        'padding': padding,
        'data_format': data_format,
        'name': name
    })

    # 4. Float32, NDHWC, VALID, stride 2
    orig_input = np.random.uniform(0, 5, (2, 4, 4, 4, 1)).astype(np.float32)
    orig_output = np.random.uniform(0, 5, (2, 2, 2, 2, 1)).astype(np.float32)
    grad = np.random.uniform(-1, 1, (2, 2, 2, 2, 1)).astype(np.float32)
    ksize = [1, 2, 2, 2, 1]
    strides = [1, 2, 2, 2, 1]
    padding = "VALID"
    data_format = "NDHWC"
    name = "max_pool_3d_grad_4"

    list_of_inputs.append({
        'orig_input': orig_input,
        'orig_output': orig_output,
        'grad': grad,
        'ksize': ksize,
        'strides': strides,
        'padding': padding,
        'data_format': data_format,
        'name': name
    })

    # 5. Float32, NDHWC, SAME, ksize 3
    orig_input = np.random.uniform(-5, 5, (1, 5, 5, 5, 3)).astype(np.float32)
    orig_output = np.random.uniform(-5, 5, (1, 5, 5, 5, 3)).astype(np.float32)
    grad = np.random.uniform(-1, 1, (1, 5, 5, 5, 3)).astype(np.float32)
    ksize = [1, 3, 3, 3, 1]
    strides = [1, 1, 1, 1, 1]
    padding = "SAME"
    data_format = "NDHWC"
    name = "max_pool_3d_grad_5"

    list_of_inputs.append({
        'orig_input': orig_input,
        'orig_output': orig_output,
        'grad': grad,
        'ksize': ksize,
        'strides': strides,
        'padding': padding,
        'data_format': data_format,
        'name': name
    })

    # 6. Float32, NDHWC, SAME, stride 2
    orig_input = np.random.uniform(-2, 2, (1, 6, 6, 6, 1)).astype(np.float32)
    orig_output = np.random.uniform(-2, 2, (1, 3, 3, 3, 1)).astype(np.float32)
    grad = np.random.uniform(-1, 1, (1, 3, 3, 3, 1)).astype(np.float32)
    ksize = [1, 2, 2, 2, 1]
    strides = [1, 2, 2, 2, 1]
    padding = "SAME"
    data_format = "NDHWC"
    name = "max_pool_3d_grad_6"

    list_of_inputs.append({
        'orig_input': orig_input,
        'orig_output': orig_output,
        'grad': grad,
        'ksize': ksize,
        'strides': strides,
        'padding': padding,
        'data_format': data_format,
        'name': name
    })

    # 7. Float32, NDHWC, VALID, ksize 3
    orig_input = np.random.uniform(-1, 1, (1, 4, 4, 4, 2)).astype(np.float32)
    orig_output = np.random.uniform(-1, 1, (1, 2, 2, 2, 2)).astype(np.float32)
    grad = np.random.uniform(-1, 1, (1, 2, 2, 2, 2)).astype(np.float32)
    ksize = [1, 3, 3, 3, 1]
    strides = [1, 1, 1, 1, 1]
    padding = "VALID"
    data_format = "NDHWC"
    name = "max_pool_3d_grad_7"

    list_of_inputs.append({
        'orig_input': orig_input,
        'orig_output': orig_output,
        'grad': grad,
        'ksize': ksize,
        'strides': strides,
        'padding': padding,
        'data_format': data_format,
        'name': name
    })

    # 8. Float32, NDHWC, VALID, stride 2, larger input
    orig_input = np.random.uniform(-1, 1, (1, 8, 8, 8, 1)).astype(np.float32)
    orig_output = np.random.uniform(-1, 1, (1, 4, 4, 4, 1)).astype(np.float32)
    grad = np.random.uniform(-1, 1, (1, 4, 4, 4, 1)).astype(np.float32)
    ksize = [1, 2, 2, 2, 1]
    strides = [1, 2, 2, 2, 1]
    padding = "VALID"
    data_format = "NDHWC"
    name = "max_pool_3d_grad_8"

    list_of_inputs.append({
        'orig_input': orig_input,
        'orig_output': orig_output,
        'grad': grad,
        'ksize': ksize,
        'strides': strides,
        'padding': padding,
        'data_format': data_format,
        'name': name
    })

    # 9. Float32, NDHWC, SAME, ksize 1, stride 1 (effectively identity)
    orig_input = np.random.uniform(-5, 5, (2, 3, 3, 3, 2)).astype(np.float32)
    orig_output = orig_input
    grad = np.random.uniform(-1, 1, (2, 3, 3, 3, 2)).astype(np.float32)
    ksize = [1, 1, 1, 1, 1]
    strides = [1, 1, 1, 1, 1]
    padding = "SAME"
    data_format = "NDHWC"
    name = "max_pool_3d_grad_9"

    list_of_inputs.append({
        'orig_input': orig_input,
        'orig_output': orig_output,
        'grad': grad,
        'ksize': ksize,
        'strides': strides,
        'padding': padding,
        'data_format': data_format,
        'name': name
    })

    # 10. Float32, NDHWC, VALID, non-symmetric input shape
    orig_input = np.random.uniform(-10, 10, (1, 4, 2, 2, 1)).astype(np.float32)
    orig_output = np.random.uniform(-10, 10, (1, 2, 1, 1, 1)).astype(np.float32)
    grad = np.random.uniform(-1, 1, (1, 2, 1, 1, 1)).astype(np.float32)
    ksize = [1, 2, 2, 2, 1]
    strides = [1, 2, 2, 2, 1]
    padding = "VALID"
    data_format = "NDHWC"
    name = "max_pool_3d_grad_10"

    list_of_inputs.append({
        'orig_input': orig_input,
        'orig_output': orig_output,
        'grad': grad,
        'ksize': ksize,
        'strides': strides,
        'padding': padding,
        'data_format': data_format,
        'name': name
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.MaxPool3DGrad"] = tf_maxpool3dgrad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.MaxPool3DGrad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MaxPool3DGrad'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.MaxPool3DGrad', generated_inputs['tf.raw_ops.MaxPool3DGrad'], lib="tf", suffix=0)
