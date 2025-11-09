
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_strided_slice_inputs():
    list_of_inputs = []

    # Input 1: basic 1D stride
    input_ = np.arange(10, dtype=np.int32)
    begin = np.array([2], dtype=np.int32)
    end = np.array([8], dtype=np.int32)
    strides = np.array([2], dtype=np.int32)
    input_dict = {
        "input_": input_,
        "begin": begin,
        "end": end,
        "strides": strides,
        "begin_mask": 0,
        "end_mask": 0,
        "ellipsis_mask": 0,
        "new_axis_mask": 0,
        "shrink_axis_mask": 0,
        "var": input_.copy(),
        "name": "basic_1d_stride2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D with column step
    input_ = np.arange(20, dtype=np.float32).reshape(4, 5)
    begin = np.array([1, 0], dtype=np.int64)
    end = np.array([4, 5], dtype=np.int64)
    strides = np.array([1, 2], dtype=np.int64)
    input_dict = {
        "input_": input_,
        "begin": begin,
        "end": end,
        "strides": strides,
        "begin_mask": 0,
        "end_mask": 0,
        "ellipsis_mask": 0,
        "new_axis_mask": 0,
        "shrink_axis_mask": 0,
        "var": input_.copy(),
        "name": "2d_step2_cols"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: reverse 1D using negative stride
    input_ = np.arange(8, dtype=np.int64)
    begin = np.array([-1], dtype=np.int64)
    end = np.array([-9], dtype=np.int64)
    strides = np.array([-1], dtype=np.int64)
    input_dict = {
        "input_": input_,
        "begin": begin,
        "end": end,
        "strides": strides,
        "begin_mask": 0,
        "end_mask": 0,
        "ellipsis_mask": 0,
        "new_axis_mask": 0,
        "shrink_axis_mask": 0,
        "var": input_.copy(),
        "name": "reverse_1d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D with begin_mask and end_mask
    input_ = np.arange(3 * 4 * 5, dtype=np.int32).reshape(3, 4, 5)
    begin = np.array([0, 1, 0], dtype=np.int32)
    end = np.array([0, 4, 5], dtype=np.int32)
    strides = np.array([1, 1, 2], dtype=np.int32)
    input_dict = {
        "input_": input_,
        "begin": begin,
        "end": end,
        "strides": strides,
        "begin_mask": 1,  # ignore begin[0]
        "end_mask": 1,    # ignore end[0]
        "ellipsis_mask": 0,
        "new_axis_mask": 0,
        "shrink_axis_mask": 0,
        "var": np.zeros_like(input_),
        "name": "3d_begin_end_mask"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D with ellipsis in the middle
    input_ = np.arange(2 * 3 * 4 * 5, dtype=np.float64).reshape(2, 3, 4, 5)
    begin = np.array([0, 1, 0], dtype=np.int32)
    end = np.array([2, 3, 5], dtype=np.int32)
    strides = np.array([1, 1, 1], dtype=np.int32)
    input_dict = {
        "input_": input_,
        "begin": begin,
        "end": end,
        "strides": strides,
        "begin_mask": 0,
        "end_mask": 0,
        "ellipsis_mask": 1 << 1,  # ellipsis at spec index 1
        "new_axis_mask": 0,
        "shrink_axis_mask": 0,
        "var": input_.copy(),
        "name": "4d_ellipsis_middle"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: new axis inserted between dims
    input_ = np.arange(12, dtype=np.int16).reshape(3, 4)
    begin = np.array([0, 0, 0], dtype=np.int64)
    end = np.array([3, 0, 4], dtype=np.int64)
    strides = np.array([1, 1, 1], dtype=np.int64)
    input_dict = {
        "input_": input_,
        "begin": begin,
        "end": end,
        "strides": strides,
        "begin_mask": 0,
        "end_mask": 0,
        "ellipsis_mask": 0,
        "new_axis_mask": 1 << 1,  # add new axis at spec index 1
        "shrink_axis_mask": 0,
        "var": input_.copy(),
        "name": "new_axis_between"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: shrink axis in the middle
    input_ = (np.arange(3 * 4 * 5, dtype=np.float16).reshape(3, 4, 5))
    begin = np.array([0, 2, 0], dtype=np.int32)
    end = np.array([0, 0, 0], dtype=np.int32)
    strides = np.array([1, 1, 1], dtype=np.int32)
    input_dict = {
        "input_": input_,
        "begin": begin,
        "end": end,
        "strides": strides,
        "begin_mask": 0,
        "end_mask": 0,
        "ellipsis_mask": 0,
        "new_axis_mask": 0,
        "shrink_axis_mask": 1 << 1,  # shrink spec index 1
        "var": input_.copy(),
        "name": "shrink_axis_mid"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D with negative stride on middle dim
    input_ = np.arange(5 * 6 * 7, dtype=np.int32).reshape(5, 6, 7)
    begin = np.array([1, 5, 0], dtype=np.int64)
    end = np.array([4, 1, 7], dtype=np.int64)
    strides = np.array([1, -2, 1], dtype=np.int64)
    input_dict = {
        "input_": input_,
        "begin": begin,
        "end": end,
        "strides": strides,
        "begin_mask": 0,
        "end_mask": 0,
        "ellipsis_mask": 0,
        "new_axis_mask": 0,
        "shrink_axis_mask": 0,
        "var": input_.copy(),
        "name": "3d_negative_stride_middle"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: shrink first spec and add two new axes
    input_ = np.linspace(0.0, 1.0, 6, dtype=np.float32)
    begin = np.array([2, 0, 0], dtype=np.int32)
    end = np.array([0, 0, 0], dtype=np.int32)
    strides = np.array([1, 1, 1], dtype=np.int32)
    input_dict = {
        "input_": input_,
        "begin": begin,
        "end": end,
        "strides": strides,
        "begin_mask": 0,
        "end_mask": 0,
        "ellipsis_mask": 0,
        "new_axis_mask": (1 << 1) | (1 << 2),  # new axes at specs 1 and 2
        "shrink_axis_mask": 1 << 0,            # shrink spec 0
        "var": input_.copy(),
        "name": "shrink_then_newaxes"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: ellipsis at first spec on 5D tensor
    input_ = np.arange(2 * 3 * 4 * 5 * 6, dtype=np.int8).reshape(2, 3, 4, 5, 6)
    begin = np.array([0, 1], dtype=np.int64)
    end = np.array([0, 5], dtype=np.int64)
    strides = np.array([1, 2], dtype=np.int64)
    input_dict = {
        "input_": input_,
        "begin": begin,
        "end": end,
        "strides": strides,
        "begin_mask": 0,
        "end_mask": 0,
        "ellipsis_mask": 1 << 0,  # ellipsis at spec index 0
        "new_axis_mask": 0,
        "shrink_axis_mask": 0,
        "var": input_.copy(),
        "name": "ellipsis_first"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: boolean 2D with strides
    input_ = (np.arange(9).reshape(3, 3) % 2 == 0)
    begin = np.array([0, 0], dtype=np.int32)
    end = np.array([3, 3], dtype=np.int32)
    strides = np.array([2, 1], dtype=np.int32)
    input_dict = {
        "input_": input_,
        "begin": begin,
        "end": end,
        "strides": strides,
        "begin_mask": 0,
        "end_mask": 0,
        "ellipsis_mask": 0,
        "new_axis_mask": 0,
        "shrink_axis_mask": 0,
        "var": input_.copy(),
        "name": "bool_2d_stride"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: 2D with negative indices
    input_ = np.arange(7 * 8, dtype=np.int32).reshape(7, 8)
    begin = np.array([5, -8], dtype=np.int64)
    end = np.array([7, -3], dtype=np.int64)
    strides = np.array([1, 1], dtype=np.int64)
    input_dict = {
        "input_": input_,
        "begin": begin,
        "end": end,
        "strides": strides,
        "begin_mask": 0,
        "end_mask": 0,
        "ellipsis_mask": 0,
        "new_axis_mask": 0,
        "shrink_axis_mask": 0,
        "var": input_.copy(),
        "name": "negative_indices_2d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.strided_slice"] = tf_strided_slice_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.strided_slice' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strided_slice'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.strided_slice', generated_inputs['tf.strided_slice'], lib="tf", suffix=0)
