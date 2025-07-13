
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Conv2DBackpropFilter_inputs():
    list_of_inputs = []

    # Input 1
    input_val = np.random.rand(1, 5, 5, 3).astype(np.float32)
    filter_sizes_val = np.array([3, 3, 3, 16], dtype=np.int32)
    out_backprop_val = np.random.rand(1, 3, 3, 16).astype(np.float32)
    strides_val = [1, 1, 1, 1]
    padding_val = "VALID"
    use_cudnn_on_gpu_val = True
    explicit_paddings_val = []
    data_format_val = "NHWC"
    dilations_val = [1, 1, 1, 1]
    name_val = "conv2d_backprop_filter_1"

    input_dict = {
        "input": input_val,
        "filter_sizes": filter_sizes_val,
        "out_backprop": out_backprop_val,
        "strides": strides_val,
        "padding": padding_val,
        "use_cudnn_on_gpu": use_cudnn_on_gpu_val,
        "explicit_paddings": explicit_paddings_val,
        "data_format": data_format_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_val = np.random.rand(2, 10, 10, 1).astype(np.float32)
    filter_sizes_val = np.array([5, 5, 1, 8], dtype=np.int32)
    out_backprop_val = np.random.rand(2, 6, 6, 8).astype(np.float32)
    strides_val = [1, 1, 1, 1]
    padding_val = "VALID"
    use_cudnn_on_gpu_val = False
    explicit_paddings_val = []
    data_format_val = "NHWC"
    dilations_val = [1, 1, 1, 1]
    name_val = "conv2d_backprop_filter_2"

    input_dict = {
        "input": input_val,
        "filter_sizes": filter_sizes_val,
        "out_backprop": out_backprop_val,
        "strides": strides_val,
        "padding": padding_val,
        "use_cudnn_on_gpu": use_cudnn_on_gpu_val,
        "explicit_paddings": explicit_paddings_val,
        "data_format": data_format_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_val = np.random.rand(1, 7, 7, 32).astype(np.float32)
    filter_sizes_val = np.array([3, 3, 32, 64], dtype=np.int32)
    out_backprop_val = np.random.rand(1, 7, 7, 64).astype(np.float32)
    strides_val = [1, 1, 1, 1]
    padding_val = "SAME"
    use_cudnn_on_gpu_val = True
    explicit_paddings_val = []
    data_format_val = "NHWC"
    dilations_val = [1, 1, 1, 1]
    name_val = "conv2d_backprop_filter_3"

    input_dict = {
        "input": input_val,
        "filter_sizes": filter_sizes_val,
        "out_backprop": out_backprop_val,
        "strides": strides_val,
        "padding": padding_val,
        "use_cudnn_on_gpu": use_cudnn_on_gpu_val,
        "explicit_paddings": explicit_paddings_val,
        "data_format": data_format_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_val = np.random.rand(4, 12, 12, 48).astype(np.float32)
    filter_sizes_val = np.array([4, 4, 48, 96], dtype=np.int32)
    out_backprop_val = np.random.rand(4, 6, 6, 96).astype(np.float32)
    strides_val = [1, 2, 2, 1]
    padding_val = "VALID"
    use_cudnn_on_gpu_val = True
    explicit_paddings_val = []
    data_format_val = "NHWC"
    dilations_val = [1, 1, 1, 1]
    name_val = "conv2d_backprop_filter_4"

    input_dict = {
        "input": input_val,
        "filter_sizes": filter_sizes_val,
        "out_backprop": out_backprop_val,
        "strides": strides_val,
        "padding": padding_val,
        "use_cudnn_on_gpu": use_cudnn_on_gpu_val,
        "explicit_paddings": explicit_paddings_val,
        "data_format": data_format_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5
    input_val = np.random.rand(1, 3, 3, 1).astype(np.float32)
    filter_sizes_val = np.array([2, 2, 1, 1], dtype=np.int32)
    out_backprop_val = np.random.rand(1, 2, 2, 1).astype(np.float32)
    strides_val = [1, 1, 1, 1]
    padding_val = "VALID"
    use_cudnn_on_gpu_val = True
    explicit_paddings_val = []
    data_format_val = "NHWC"
    dilations_val = [1, 1, 1, 1]
    name_val = "conv2d_backprop_filter_5"

    input_dict = {
        "input": input_val,
        "filter_sizes": filter_sizes_val,
        "out_backprop": out_backprop_val,
        "strides": strides_val,
        "padding": padding_val,
        "use_cudnn_on_gpu": use_cudnn_on_gpu_val,
        "explicit_paddings": explicit_paddings_val,
        "data_format": data_format_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_val = np.random.rand(1, 5, 5, 3).astype(np.float32)
    filter_sizes_val = np.array([3, 3, 3, 16], dtype=np.int32)
    out_backprop_val = np.random.rand(1, 7, 7, 16).astype(np.float32)
    strides_val = [1, 1, 1, 1]
    padding_val = "SAME"
    use_cudnn_on_gpu_val = True
    explicit_paddings_val = []
    data_format_val = "NHWC"
    dilations_val = [1, 1, 1, 1]
    name_val = "conv2d_backprop_filter_6"

    input_dict = {
        "input": input_val,
        "filter_sizes": filter_sizes_val,
        "out_backprop": out_backprop_val,
        "strides": strides_val,
        "padding": padding_val,
        "use_cudnn_on_gpu": use_cudnn_on_gpu_val,
        "explicit_paddings": explicit_paddings_val,
        "data_format": data_format_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_val = np.random.rand(1, 5, 5, 3).astype(np.float32)
    filter_sizes_val = np.array([3, 3, 3, 16], dtype=np.int32)
    out_backprop_val = np.random.rand(1, 3, 3, 16).astype(np.float32)
    strides_val = [1, 1, 1, 1]
    padding_val = "EXPLICIT"
    use_cudnn_on_gpu_val = True
    explicit_paddings_val = [0, 0, 1, 1, 2, 2, 0, 0]
    data_format_val = "NHWC"
    dilations_val = [1, 1, 1, 1]
    name_val = "conv2d_backprop_filter_7"

    input_dict = {
        "input": input_val,
        "filter_sizes": filter_sizes_val,
        "out_backprop": out_backprop_val,
        "strides": strides_val,
        "padding": padding_val,
        "use_cudnn_on_gpu": use_cudnn_on_gpu_val,
        "explicit_paddings": explicit_paddings_val,
        "data_format": data_format_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_val = np.random.rand(1, 5, 5, 3).astype(np.float32)
    filter_sizes_val = np.array([3, 3, 3, 16], dtype=np.int32)
    out_backprop_val = np.random.rand(1, 3, 3, 16).astype(np.float32)
    strides_val = [1, 1, 1, 1]
    padding_val = "VALID"
    use_cudnn_on_gpu_val = True
    explicit_paddings_val = []
    data_format_val = "NCHW"
    dilations_val = [1, 1, 1, 1]
    name_val = "conv2d_backprop_filter_8"

    input_dict = {
        "input": input_val,
        "filter_sizes": filter_sizes_val,
        "out_backprop": out_backprop_val,
        "strides": strides_val,
        "padding": padding_val,
        "use_cudnn_on_gpu": use_cudnn_on_gpu_val,
        "explicit_paddings": explicit_paddings_val,
        "data_format": data_format_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_val = np.random.rand(1, 5, 5, 3).astype(np.float32)
    filter_sizes_val = np.array([3, 3, 3, 16], dtype=np.int32)
    out_backprop_val = np.random.rand(1, 3, 3, 16).astype(np.float32)
    strides_val = [1, 1, 1, 1]
    padding_val = "VALID"
    use_cudnn_on_gpu_val = True
    explicit_paddings_val = []
    data_format_val = "NHWC"
    dilations_val = [1, 2, 2, 1]
    name_val = "conv2d_backprop_filter_9"

    input_dict = {
        "input": input_val,
        "filter_sizes": filter_sizes_val,
        "out_backprop": out_backprop_val,
        "strides": strides_val,
        "padding": padding_val,
        "use_cudnn_on_gpu": use_cudnn_on_gpu_val,
        "explicit_paddings": explicit_paddings_val,
        "data_format": data_format_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: bfloat16
    input_val = np.random.rand(1, 5, 5, 3).astype(np.float16)
    filter_sizes_val = np.array([3, 3, 3, 16], dtype=np.int32)
    out_backprop_val = np.random.rand(1, 3, 3, 16).astype(np.float16)
    strides_val = [1, 1, 1, 1]
    padding_val = "VALID"
    use_cudnn_on_gpu_val = True
    explicit_paddings_val = []
    data_format_val = "NHWC"
    dilations_val = [1, 1, 1, 1]
    name_val = "conv2d_backprop_filter_10"

    input_dict = {
        "input": input_val,
        "filter_sizes": filter_sizes_val,
        "out_backprop": out_backprop_val,
        "strides": strides_val,
        "padding": padding_val,
        "use_cudnn_on_gpu": use_cudnn_on_gpu_val,
        "explicit_paddings": explicit_paddings_val,
        "data_format": data_format_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_Conv2DBackpropFilter_inputs()
generated_inputs["tf.raw_ops.Conv2DBackpropFilter"] = inputs
for i in range(len(generated_inputs["tf.raw_ops.Conv2DBackpropFilter"])):
    for key in generated_inputs["tf.raw_ops.Conv2DBackpropFilter"][i]:
        if isinstance(generated_inputs["tf.raw_ops.Conv2DBackpropFilter"][i][key], np.ndarray):
            generated_inputs["tf.raw_ops.Conv2DBackpropFilter"][i][key] = tf.convert_to_tensor(generated_inputs["tf.raw_ops.Conv2DBackpropFilter"][i][key])
        if key == "strides" or key == "explicit_paddings" or key == "dilations":
            generated_inputs["tf.raw_ops.Conv2DBackpropFilter"][i][key] = tuple(generated_inputs["tf.raw_ops.Conv2DBackpropFilter"][i][key])

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Conv2DBackpropFilter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Conv2DBackpropFilter'.")

check_valid('tf.raw_ops.Conv2DBackpropFilter', generated_inputs['tf.raw_ops.Conv2DBackpropFilter'], lib="tf", suffix=0)
