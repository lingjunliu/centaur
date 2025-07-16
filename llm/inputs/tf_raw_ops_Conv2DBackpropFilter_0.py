
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Conv2DBackpropFilter_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.rand(1, 5, 5, 3).astype(np.float32)
    filter_sizes_tensor = np.array([3, 3, 3, 16], dtype=np.int32)
    out_backprop_tensor = np.random.rand(1, 3, 3, 16).astype(np.float32)
    strides_list = [1, 1, 1, 1]
    padding_string = "VALID"
    use_cudnn_on_gpu_bool = True
    explicit_paddings_list = []
    data_format_string = "NHWC"
    dilations_list = [1, 1, 1, 1]
    name_string = "conv2d_backprop_filter_1"

    input_dict = {
        "input": input_tensor,
        "filter_sizes": filter_sizes_tensor,
        "out_backprop": out_backprop_tensor,
        "strides": strides_list,
        "padding": padding_string,
        "use_cudnn_on_gpu": use_cudnn_on_gpu_bool,
        "explicit_paddings": explicit_paddings_list,
        "data_format": data_format_string,
        "dilations": dilations_list,
        "name": name_string
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2 (Adjusted out_backprop shape for VALID padding and strides)
    input_tensor = np.random.rand(2, 10, 10, 1).astype(np.float64)
    filter_sizes_tensor = np.array([5, 5, 1, 8], dtype=np.int32)
    out_backprop_tensor = np.random.rand(2, 3, 3, 8).astype(np.float64) # Adjusted shape
    strides_list = [1, 2, 2, 1]
    padding_string = "VALID"
    use_cudnn_on_gpu_bool = False
    explicit_paddings_list = []
    data_format_string = "NHWC"
    dilations_list = [1, 1, 1, 1]
    name_string = "conv2d_backprop_filter_2"

    input_dict = {
        "input": input_tensor,
        "filter_sizes": filter_sizes_tensor,
        "out_backprop": out_backprop_tensor,
        "strides": strides_list,
        "padding": padding_string,
        "use_cudnn_on_gpu": use_cudnn_on_gpu_bool,
        "explicit_paddings": explicit_paddings_list,
        "data_format": data_format_string,
        "dilations": dilations_list,
        "name": name_string
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.rand(1, 7, 7, 3).astype(np.float32)
    filter_sizes_tensor = np.array([3, 3, 3, 1], dtype=np.int32)
    out_backprop_tensor = np.random.rand(1, 7, 7, 1).astype(np.float32)
    strides_list = [1, 1, 1, 1]
    padding_string = "SAME"
    use_cudnn_on_gpu_bool = True
    explicit_paddings_list = []
    data_format_string = "NHWC"
    dilations_list = [1, 1, 1, 1]
    name_string = "conv2d_backprop_filter_3"

    input_dict = {
        "input": input_tensor,
        "filter_sizes": filter_sizes_tensor,
        "out_backprop": out_backprop_tensor,
        "strides": strides_list,
        "padding": padding_string,
        "use_cudnn_on_gpu": use_cudnn_on_gpu_bool,
        "explicit_paddings": explicit_paddings_list,
        "data_format": data_format_string,
        "dilations": dilations_list,
        "name": name_string
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.random.rand(4, 12, 12, 5).astype(np.float32)
    filter_sizes_tensor = np.array([2, 2, 5, 10], dtype=np.int32)
    out_backprop_tensor = np.random.rand(4, 12, 12, 10).astype(np.float32)
    strides_list = [1, 1, 1, 1]
    padding_string = "SAME"
    use_cudnn_on_gpu_bool = False
    explicit_paddings_list = []
    data_format_string = "NHWC"
    dilations_list = [1, 1, 1, 1]
    name_string = "conv2d_backprop_filter_4"

    input_dict = {
        "input": input_tensor,
        "filter_sizes": filter_sizes_tensor,
        "out_backprop": out_backprop_tensor,
        "strides": strides_list,
        "padding": padding_string,
        "use_cudnn_on_gpu": use_cudnn_on_gpu_bool,
        "explicit_paddings": explicit_paddings_list,
        "data_format": data_format_string,
        "dilations": dilations_list,
        "name": name_string
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.random.rand(1, 8, 8, 3).astype(np.float32)
    filter_sizes_tensor = np.array([3, 3, 3, 1], dtype=np.int32)
    out_backprop_tensor = np.random.rand(1, 3, 3, 1).astype(np.float32)  # Adjusted shape
    strides_list = [1, 2, 2, 1]
    padding_string = "VALID"
    use_cudnn_on_gpu_bool = True
    explicit_paddings_list = []
    data_format_string = "NHWC"
    dilations_list = [1, 1, 1, 1]
    name_string = "conv2d_backprop_filter_5"

    input_dict = {
        "input": input_tensor,
        "filter_sizes": filter_sizes_tensor,
        "out_backprop": out_backprop_tensor,
        "strides": strides_list,
        "padding": padding_string,
        "use_cudnn_on_gpu": use_cudnn_on_gpu_bool,
        "explicit_paddings": explicit_paddings_list,
        "data_format": data_format_string,
        "dilations": dilations_list,
        "name": name_string
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.random.rand(1, 8, 8, 3).astype(np.float32)
    filter_sizes_tensor = np.array([3, 3, 3, 1], dtype=np.int32)
    out_backprop_tensor = np.random.rand(1, 8, 8, 1).astype(np.float32)
    strides_list = [1, 1, 1, 1]
    padding_string = "SAME"
    use_cudnn_on_gpu_bool = True
    explicit_paddings_list = []
    data_format_string = "NHWC"
    dilations_list = [1, 2, 2, 1]
    name_string = "conv2d_backprop_filter_6"

    input_dict = {
        "input": input_tensor,
        "filter_sizes": filter_sizes_tensor,
        "out_backprop": out_backprop_tensor,
        "strides": strides_list,
        "padding": padding_string,
        "use_cudnn_on_gpu": use_cudnn_on_gpu_bool,
        "explicit_paddings": explicit_paddings_list,
        "data_format": data_format_string,
        "dilations": dilations_list,
        "name": name_string
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 (Adjusted out_backprop shape for EXPLICIT padding, and removed batch/depth padding)
    input_tensor = np.random.rand(1, 5, 5, 3).astype(np.float32)
    filter_sizes_tensor = np.array([3, 3, 3, 16], dtype=np.int32)
    strides_list = [1, 1, 1, 1]
    padding_string = "EXPLICIT"
    use_cudnn_on_gpu_bool = True
    explicit_paddings_list = [0, 0, 1, 1, 1, 1, 0, 0]  # Adjusted Paddings, only spatial
    data_format_string = "NHWC"
    dilations_list = [1, 1, 1, 1]

    # Calculate expected output shape based on explicit padding
    in_height = input_tensor.shape[1]
    in_width = input_tensor.shape[2]
    filter_height = filter_sizes_tensor[0]
    filter_width = filter_sizes_tensor[1]
    pad_top = explicit_paddings_list[2]
    pad_bottom = explicit_paddings_list[3]
    pad_left = explicit_paddings_list[4]
    pad_right = explicit_paddings_list[5]
    stride_height = strides_list[1]
    stride_width = strides_list[2]
    dilation_height = dilations_list[1]
    dilation_width = dilations_list[2]

    out_height = int(((in_height + pad_top + pad_bottom - dilation_height * (filter_height - 1) - 1) / stride_height) + 1)
    out_width = int(((in_width + pad_left + pad_right - dilation_width * (filter_width - 1) - 1) / stride_width) + 1)

    out_backprop_tensor = np.random.rand(1, out_height, out_width, 16).astype(np.float32)
    name_string = "conv2d_backprop_filter_7"


    input_dict = {
        "input": input_tensor,
        "filter_sizes": filter_sizes_tensor,
        "out_backprop": out_backprop_tensor,
        "strides": strides_list,
        "padding": padding_string,
        "use_cudnn_on_gpu": use_cudnn_on_gpu_bool,
        "explicit_paddings": explicit_paddings_list,
        "data_format": data_format_string,
        "dilations": dilations_list,
        "name": name_string
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8: NCHW format
    input_tensor = np.random.rand(2, 3, 10, 10).astype(np.float32)
    filter_sizes_tensor = np.array([3, 3, 3, 16], dtype=np.int32)
    out_backprop_tensor = np.random.rand(2, 16, 8, 8).astype(np.float32)
    strides_list = [1, 1, 1, 1]
    padding_string = "VALID"
    use_cudnn_on_gpu_bool = True
    explicit_paddings_list = []
    data_format_string = "NCHW"
    dilations_list = [1, 1, 1, 1]
    name_string = "conv2d_backprop_filter_8"

    input_dict = {
        "input": input_tensor,
        "filter_sizes": filter_sizes_tensor,
        "out_backprop": out_backprop_tensor,
        "strides": strides_list,
        "padding": padding_string,
        "use_cudnn_on_gpu": use_cudnn_on_gpu_bool,
        "explicit_paddings": explicit_paddings_list,
        "data_format": data_format_string,
        "dilations": dilations_list,
        "name": name_string
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.random.rand(1, 5, 5, 3).astype(np.float16)
    filter_sizes_tensor = np.array([3, 3, 3, 16], dtype=np.int32)
    out_backprop_tensor = np.random.rand(1, 3, 3, 16).astype(np.float16)
    strides_list = [1, 1, 1, 1]
    padding_string = "VALID"
    use_cudnn_on_gpu_bool = True
    explicit_paddings_list = []
    data_format_string = "NHWC"
    dilations_list = [1, 1, 1, 1]
    name_string = "conv2d_backprop_filter_9"

    input_dict = {
        "input": input_tensor,
        "filter_sizes": filter_sizes_tensor,
        "out_backprop": out_backprop_tensor,
        "strides": strides_list,
        "padding": padding_string,
        "use_cudnn_on_gpu": use_cudnn_on_gpu_bool,
        "explicit_paddings": explicit_paddings_list,
        "data_format": data_format_string,
        "dilations": dilations_list,
        "name": name_string
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Dilation
    input_tensor = np.random.rand(1, 8, 8, 3).astype(np.float32)
    filter_sizes_tensor = np.array([3, 3, 3, 1], dtype=np.int32)
    out_backprop_tensor = np.random.rand(1, 6, 6, 1).astype(np.float32)
    strides_list = [1, 1, 1, 1]
    padding_string = "VALID"
    use_cudnn_on_gpu_bool = True
    explicit_paddings_list = []
    data_format_string = "NHWC"
    dilations_list = [1, 2, 2, 1]
    name_string = "conv2d_backprop_filter_10"

    input_dict = {
        "input": input_tensor,
        "filter_sizes": filter_sizes_tensor,
        "out_backprop": out_backprop_tensor,
        "strides": strides_list,
        "padding": padding_string,
        "use_cudnn_on_gpu": use_cudnn_on_gpu_bool,
        "explicit_paddings": explicit_paddings_list,
        "data_format": data_format_string,
        "dilations": dilations_list,
        "name": name_string
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Conv2DBackpropFilter"] = tf_raw_ops_Conv2DBackpropFilter_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Conv2DBackpropFilter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Conv2DBackpropFilter'.")

check_valid('tf.raw_ops.Conv2DBackpropFilter', generated_inputs['tf.raw_ops.Conv2DBackpropFilter'], lib="tf", suffix=0)
