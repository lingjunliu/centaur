
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import math

def get_dilation2dbackpropfilter_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.Dilation2DBackpropFilter function.
    """
    list_of_inputs = []

    def get_out_size(in_size, f_size, rate, stride, padding):
        if padding.upper() == 'VALID':
            return int(math.ceil((in_size - (f_size - 1) * rate) / stride))
        elif padding.upper() == 'SAME':
            return int(math.ceil(in_size / stride))
        return 0

    # Input 1: Basic case, float32, VALID padding
    in_shape = [1, 5, 5, 1]
    f_shape = [3, 3, 1]
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"
    out_h = get_out_size(in_shape[1], f_shape[0], rates[1], strides[1], padding)
    out_w = get_out_size(in_shape[2], f_shape[1], rates[2], strides[2], padding)
    out_backprop_shape = [in_shape[0], out_h, out_w, in_shape[3]]
    list_of_inputs.append({
        "name": "basic_valid_float32",
        "input": np.random.rand(*in_shape).astype(np.float32),
        "filter": np.random.rand(*f_shape).astype(np.float32),
        "out_backprop": np.random.rand(*out_backprop_shape).astype(np.float32),
        "strides": strides,
        "rates": rates,
        "padding": padding
    })

    # Input 2: Basic case, float32, SAME padding
    padding = "SAME"
    out_h = get_out_size(in_shape[1], f_shape[0], rates[1], strides[1], padding)
    out_w = get_out_size(in_shape[2], f_shape[1], rates[2], strides[2], padding)
    out_backprop_shape = [in_shape[0], out_h, out_w, in_shape[3]]
    list_of_inputs.append({
        "name": "basic_same_float32",
        "input": np.random.rand(*in_shape).astype(np.float32),
        "filter": np.random.rand(*f_shape).astype(np.float32),
        "out_backprop": np.random.rand(*out_backprop_shape).astype(np.float32),
        "strides": strides,
        "rates": rates,
        "padding": padding
    })

    # Input 3: Strides > 1, int32, VALID padding, with negative values
    in_shape = [2, 7, 7, 3]
    f_shape = [3, 3, 3]
    strides = [1, 2, 2, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"
    out_h = get_out_size(in_shape[1], f_shape[0], rates[1], strides[1], padding)
    out_w = get_out_size(in_shape[2], f_shape[1], rates[2], strides[2], padding)
    out_backprop_shape = [in_shape[0], out_h, out_w, in_shape[3]]
    list_of_inputs.append({
        "name": "strided_valid_int32",
        "input": np.random.randint(-100, 100, size=in_shape).astype(np.int32),
        "filter": np.random.randint(-100, 100, size=f_shape).astype(np.int32),
        "out_backprop": np.random.randint(-100, 100, size=out_backprop_shape).astype(np.int32),
        "strides": strides,
        "rates": rates,
        "padding": padding
    })

    # Input 4: Strides > 1, int32, SAME padding
    padding = "SAME"
    out_h = get_out_size(in_shape[1], f_shape[0], rates[1], strides[1], padding)
    out_w = get_out_size(in_shape[2], f_shape[1], rates[2], strides[2], padding)
    out_backprop_shape = [in_shape[0], out_h, out_w, in_shape[3]]
    list_of_inputs.append({
        "name": "strided_same_int32",
        "input": np.random.randint(-50, 50, size=in_shape).astype(np.int32),
        "filter": np.random.randint(-50, 50, size=f_shape).astype(np.int32),
        "out_backprop": np.random.randint(-50, 50, size=out_backprop_shape).astype(np.int32),
        "strides": strides,
        "rates": rates,
        "padding": padding
    })

    # Input 5: Rates > 1 (Atrous), uint8, VALID padding
    in_shape = [1, 9, 9, 1]
    f_shape = [3, 3, 1]
    strides = [1, 1, 1, 1]
    rates = [1, 2, 2, 1]
    padding = "VALID"
    out_h = get_out_size(in_shape[1], f_shape[0], rates[1], strides[1], padding)
    out_w = get_out_size(in_shape[2], f_shape[1], rates[2], strides[2], padding)
    out_backprop_shape = [in_shape[0], out_h, out_w, in_shape[3]]
    list_of_inputs.append({
        "name": "rated_valid_uint8",
        "input": np.random.randint(0, 255, size=in_shape, dtype=np.uint8),
        "filter": np.random.randint(0, 255, size=f_shape, dtype=np.uint8),
        "out_backprop": np.random.randint(0, 255, size=out_backprop_shape, dtype=np.uint8),
        "strides": strides,
        "rates": rates,
        "padding": padding
    })

    # Input 6: Rates > 1 (Atrous), uint8, SAME padding
    padding = "SAME"
    out_h = get_out_size(in_shape[1], f_shape[0], rates[1], strides[1], padding)
    out_w = get_out_size(in_shape[2], f_shape[1], rates[2], strides[2], padding)
    out_backprop_shape = [in_shape[0], out_h, out_w, in_shape[3]]
    list_of_inputs.append({
        "name": "rated_same_uint8",
        "input": np.random.randint(0, 255, size=in_shape, dtype=np.uint8),
        "filter": np.random.randint(0, 255, size=f_shape, dtype=np.uint8),
        "out_backprop": np.random.randint(0, 255, size=out_backprop_shape, dtype=np.uint8),
        "strides": strides,
        "rates": rates,
        "padding": padding
    })

    # Input 7: Strides and Rates > 1, float64, VALID padding
    in_shape = [1, 10, 10, 2]
    f_shape = [4, 2, 2]
    strides = [1, 2, 1, 1]
    rates = [1, 1, 3, 1]
    padding = "VALID"
    out_h = get_out_size(in_shape[1], f_shape[0], rates[1], strides[1], padding)
    out_w = get_out_size(in_shape[2], f_shape[1], rates[2], strides[2], padding)
    out_backprop_shape = [in_shape[0], out_h, out_w, in_shape[3]]
    list_of_inputs.append({
        "name": "strided_rated_valid_float64",
        "input": np.random.rand(*in_shape).astype(np.float64) * 200 - 100,
        "filter": np.random.rand(*f_shape).astype(np.float64) * 200 - 100,
        "out_backprop": np.random.rand(*out_backprop_shape).astype(np.float64) * 200 - 100,
        "strides": strides,
        "rates": rates,
        "padding": padding
    })

    # Input 8: Strides and Rates > 1, int16, SAME padding
    in_shape = [1, 15, 15, 1]
    f_shape = [3, 3, 1]
    strides = [1, 3, 3, 1]
    rates = [1, 2, 2, 1]
    padding = "SAME"
    out_h = get_out_size(in_shape[1], f_shape[0], rates[1], strides[1], padding)
    out_w = get_out_size(in_shape[2], f_shape[1], rates[2], strides[2], padding)
    out_backprop_shape = [in_shape[0], out_h, out_w, in_shape[3]]
    list_of_inputs.append({
        "name": "strided_rated_same_int16",
        "input": np.random.randint(-30000, 30000, size=in_shape).astype(np.int16),
        "filter": np.random.randint(-30000, 30000, size=f_shape).astype(np.int16),
        "out_backprop": np.random.randint(-30000, 30000, size=out_backprop_shape).astype(np.int16),
        "strides": strides,
        "rates": rates,
        "padding": padding
    })

    # Input 9: Non-square filter, float32, VALID padding
    in_shape = [1, 8, 6, 4]
    f_shape = [3, 2, 4]
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"
    out_h = get_out_size(in_shape[1], f_shape[0], rates[1], strides[1], padding)
    out_w = get_out_size(in_shape[2], f_shape[1], rates[2], strides[2], padding)
    out_backprop_shape = [in_shape[0], out_h, out_w, in_shape[3]]
    list_of_inputs.append({
        "name": "nonsquare_filter_valid_float32",
        "input": np.random.rand(*in_shape).astype(np.float32),
        "filter": np.random.rand(*f_shape).astype(np.float32),
        "out_backprop": np.random.rand(*out_backprop_shape).astype(np.float32),
        "strides": strides,
        "rates": rates,
        "padding": padding
    })

    # Input 10: Larger batch and depth, int8, SAME padding
    in_shape = [4, 10, 10, 8]
    f_shape = [3, 3, 8]
    strides = [1, 2, 2, 1]
    rates = [1, 1, 1, 1]
    padding = "SAME"
    out_h = get_out_size(in_shape[1], f_shape[0], rates[1], strides[1], padding)
    out_w = get_out_size(in_shape[2], f_shape[1], rates[2], strides[2], padding)
    out_backprop_shape = [in_shape[0], out_h, out_w, in_shape[3]]
    list_of_inputs.append({
        "name": "large_batch_depth_same_int8",
        "input": np.random.randint(-128, 127, size=in_shape).astype(np.int8),
        "filter": np.random.randint(-128, 127, size=f_shape).astype(np.int8),
        "out_backprop": np.random.randint(-128, 127, size=out_backprop_shape).astype(np.int8),
        "strides": strides,
        "rates": rates,
        "padding": padding
    })

    # Input 11: Another dtype `half` (np.float16)
    in_shape = [1, 6, 6, 2]
    f_shape = [2, 2, 2]
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"
    out_h = get_out_size(in_shape[1], f_shape[0], rates[1], strides[1], padding)
    out_w = get_out_size(in_shape[2], f_shape[1], rates[2], strides[2], padding)
    out_backprop_shape = [in_shape[0], out_h, out_w, in_shape[3]]
    list_of_inputs.append({
        "name": "basic_valid_float16",
        "input": np.random.rand(*in_shape).astype(np.float16),
        "filter": np.random.rand(*f_shape).astype(np.float16),
        "out_backprop": np.random.rand(*out_backprop_shape).astype(np.float16),
        "strides": strides,
        "rates": rates,
        "padding": padding
    })

    # Input 12: Minimal valid case, int64
    in_shape = [1, 3, 3, 1]
    f_shape = [3, 3, 1]
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"
    out_h = get_out_size(in_shape[1], f_shape[0], rates[1], strides[1], padding)
    out_w = get_out_size(in_shape[2], f_shape[1], rates[2], strides[2], padding)
    out_backprop_shape = [in_shape[0], out_h, out_w, in_shape[3]]
    list_of_inputs.append({
        "name": "minimal_valid_int64",
        "input": np.random.randint(0, 100, size=in_shape).astype(np.int64),
        "filter": np.random.randint(0, 100, size=f_shape).astype(np.int64),
        "out_backprop": np.random.randint(0, 100, size=out_backprop_shape).astype(np.int64),
        "strides": strides,
        "rates": rates,
        "padding": padding
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.Dilation2DBackpropFilter"] = get_dilation2dbackpropfilter_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Dilation2DBackpropFilter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Dilation2DBackpropFilter'.")

check_valid('tf.raw_ops.Dilation2DBackpropFilter', generated_inputs['tf.raw_ops.Dilation2DBackpropFilter'], lib="tf", suffix=0)
