
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_MaxPoolGradV2_inputs():
    configs = [
        {
            "data_format": "NHWC", "padding": "VALID",
            "input_shape": [1, 2, 2, 1], "ksize": [1, 2, 2, 1], "strides": [1, 2, 2, 1]
        },
        {
            "data_format": "NHWC", "padding": "SAME",
            "input_shape": [1, 2, 2, 1], "ksize": [1, 2, 2, 1], "strides": [1, 1, 1, 1]
        },
        {
            "data_format": "NHWC", "padding": "VALID",
            "input_shape": [1, 3, 3, 1], "ksize": [1, 2, 2, 1], "strides": [1, 2, 2, 1]
        },
        {
            "data_format": "NHWC", "padding": "SAME",
            "input_shape": [1, 3, 3, 1], "ksize": [1, 2, 2, 1], "strides": [1, 1, 1, 1]
        },
        {
            "data_format": "NHWC", "padding": "SAME",
            "input_shape": [2, 3, 3, 2], "ksize": [1, 2, 2, 1], "strides": [1, 2, 2, 1]
        },
        {
            "data_format": "NHWC", "padding": "VALID",
            "input_shape": [2, 3, 3, 2], "ksize": [1, 2, 2, 1], "strides": [1, 1, 1, 1]
        },
        {
            "data_format": "NHWC", "padding": "SAME",
            "input_shape": [1, 4, 4, 1], "ksize": [1, 2, 2, 1], "strides": [1, 2, 2, 1]
        },
        {
            "data_format": "NHWC", "padding": "VALID",
            "input_shape": [1, 4, 4, 1], "ksize": [1, 2, 2, 1], "strides": [1, 1, 1, 1]
        },
        {
            "data_format": "NHWC", "padding": "VALID",
            "input_shape": [1, 2, 2, 3], "ksize": [1, 2, 2, 1], "strides": [1, 1, 1, 1]
        },
        {
            "data_format": "NHWC", "padding": "SAME",
            "input_shape": [2, 2, 2, 2], "ksize": [1, 1, 1, 1], "strides": [1, 1, 1, 1]
        }
    ]

    list_of_inputs = []
    
    with tf.device('/CPU:0'):
        for i, cfg in enumerate(configs):
            num_elements = int(np.prod(cfg["input_shape"]))
            arr = np.linspace(-2.0, 2.0, num_elements).astype(np.float32)
            orig_input = arr.reshape(cfg["input_shape"])
            
            ksize = cfg["ksize"]
            strides = cfg["strides"]
            padding = cfg["padding"]
            data_format = cfg["data_format"]
            
            orig_output = tf.raw_ops.MaxPool(
                input=orig_input,
                ksize=ksize,
                strides=strides,
                padding=padding,
                data_format="NHWC"
            ).numpy()
                
            orig_output = orig_output.astype(np.float32)
            grad_elements = int(np.prod(orig_output.shape))
            grad_arr = np.linspace(0.1, 1.0, grad_elements).astype(np.float32)
            grad = grad_arr.reshape(orig_output.shape)
            
            input_dict = {
                "padding": padding,
                "data_format": data_format,
                "name": f"maxpoolgradv2_{i}",
                "orig_input": orig_input,
                "orig_output": orig_output,
                "grad": grad,
                "ksize": np.array(ksize, dtype=np.int32),
                "strides": np.array(strides, dtype=np.int32)
            }
            list_of_inputs.append(copy.deepcopy(input_dict))
            
    return list_of_inputs

generated_inputs["tf.raw_ops.MaxPoolGradV2"] = tf_raw_ops_MaxPoolGradV2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.MaxPoolGradV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MaxPoolGradV2'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.MaxPoolGradV2', generated_inputs['tf.raw_ops.MaxPoolGradV2'], lib="tf", suffix=0)
