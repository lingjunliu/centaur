generated_inputs = {}

import numpy as np
import copy

def xla_add_inputs():
    list_of_inputs = []

    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([4, 5, 6], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    x = np.array([-1, -2, -3], dtype=np.int32)
    y = np.array([3, 2, 1], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    x = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    y = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    x = np.array([[1, 2], [3, 4]], dtype=np.int64)
    y = np.array([[5, 6], [7, 8]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    x = np.array([[[1]], [[2]]], dtype=np.int32)
    y = np.array([[[3]], [[4]]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    x = np.array([1.0, -1.0, 0.0], dtype=np.float64)
    y = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    x = np.array([[[-1, -2], [-3, -4]]], dtype=np.int32)
    y = np.array([[[4, 3], [2, 1]]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    x = np.array([10], dtype=np.int32)
    y = np.array([20], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    x = np.array([[1.1, 2.2, 3.3]], dtype=np.float32)
    y = np.array([[3.3, 2.2, 1.1]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    x = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.int64)
    y = np.array([[[6, 5, 4], [3, 2, 1]]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    x = np.zeros((2, 2), dtype=np.float32)
    y = np.ones((2, 2), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    x = np.array([[100, -100], [-50, 50]], dtype=np.int32)
    y = np.array([[1, 1], [1, 1]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs

generated_inputs["xla.add"] = xla_add_inputs()

import numpy as np
import copy

def xla_broadcast_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([1, 2, 3], dtype=np.int32)
    shape = (3, 3)
    list_of_inputs.append(copy.deepcopy({"x": x, "shape": shape}))

    # Input 2
    x = np.array([[1], [2], [3]], dtype=np.float32)
    shape = (3, 4)
    list_of_inputs.append(copy.deepcopy({"x": x, "shape": shape}))

    # Input 3
    x = np.array(5, dtype=np.int64)
    shape = (2, 2, 2)
    list_of_inputs.append(copy.deepcopy({"x": x, "shape": shape}))

    # Input 4
    x = np.array([[1, 2]], dtype=np.float64)
    shape = (4, 2)
    list_of_inputs.append(copy.deepcopy({"x": x, "shape": shape}))

    # Input 5
    x = np.array([[-1, -2, -3]], dtype=np.int32)
    shape = (5, 3)
    list_of_inputs.append(copy.deepcopy({"x": x, "shape": shape}))

    # Input 6
    x = np.array([[1], [2]], dtype=np.float32)
    shape = (2, 3)
    list_of_inputs.append(copy.deepcopy({"x": x, "shape": shape}))

    # Input 7
    x = np.array([1], dtype=np.int32)
    shape = (4,)
    list_of_inputs.append(copy.deepcopy({"x": x, "shape": shape}))

    # Input 8
    x = np.array([[1, 2, 3]], dtype=np.int64)
    shape = (2, 3)
    list_of_inputs.append(copy.deepcopy({"x": x, "shape": shape}))

    # Input 9
    x = np.array([[[1]], [[2]]], dtype=np.float32)
    shape = (2, 3, 4)
    list_of_inputs.append(copy.deepcopy({"x": x, "shape": shape}))

    # Input 10
    x = np.array([[1, 2], [3, 4]], dtype=np.float64)
    shape = (2, 2)
    list_of_inputs.append(copy.deepcopy({"x": x, "shape": shape}))

    # Input 11
    x = np.array([0], dtype=np.int32)
    shape = (3, 3, 3)
    list_of_inputs.append(copy.deepcopy({"x": x, "shape": shape}))

    # Input 12
    x = np.array([[True], [False]], dtype=np.bool_)
    shape = (2, 2)
    list_of_inputs.append(copy.deepcopy({"x": x, "shape": shape}))

    return list_of_inputs

generated_inputs["xla.broadcast"] = xla_broadcast_inputs()

import numpy as np
import copy

def xla_dequantize_inputs():
    list_of_inputs = []

    input_dict = {
        "input": np.array([0, 127, 255], dtype=np.uint8),
        "min_range": 0.0,
        "max_range": 255.0,
        "mode": "MIN_COMBINED",
        "transpose_output": False,
        "name": "basic_uint8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.array([-128, 0, 127], dtype=np.int8),
        "min_range": -1.0,
        "max_range": 1.0,
        "mode": "MIN_FIRST",
        "transpose_output": True,
        "name": "int8_range"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.array([[0, 10], [20, 30]], dtype=np.uint8),
        "min_range": 0.0,
        "max_range": 30.0,
        "mode": "SCALED",
        "transpose_output": False,
        "name": "2d_tensor"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.array([[255, 128], [64, 32]], dtype=np.uint8),
        "min_range": 0.0,
        "max_range": 255.0,
        "mode": "MIN_COMBINED",
        "transpose_output": True,
        "name": "transpose_case"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.array([[[0, 1], [2, 3]]], dtype=np.uint8),
        "min_range": 0.0,
        "max_range": 3.0,
        "mode": "MIN_FIRST",
        "transpose_output": False,
        "name": "3d_small"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.array([-10, 0, 10], dtype=np.int8),
        "min_range": -10.0,
        "max_range": 10.0,
        "mode": "SCALED",
        "transpose_output": False,
        "name": "negative_positive"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.array([[1, 1, 1], [1, 1, 1]], dtype=np.uint8),
        "min_range": 0.0,
        "max_range": 1.0,
        "mode": "MIN_COMBINED",
        "transpose_output": False,
        "name": "constant_values"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.array([[[255], [128]], [[64], [32]]], dtype=np.uint8),
        "min_range": 0.0,
        "max_range": 255.0,
        "mode": "MIN_FIRST",
        "transpose_output": True,
        "name": "3d_transpose"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.array([5, 10, 15, 20], dtype=np.uint8),
        "min_range": 5.0,
        "max_range": 20.0,
        "mode": "SCALED",
        "transpose_output": False,
        "name": "offset_range"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.array([[0, 255], [255, 0]], dtype=np.uint8),
        "min_range": 0.0,
        "max_range": 255.0,
        "mode": "MIN_COMBINED",
        "transpose_output": True,
        "name": "checkerboard"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["xla.dequantize"] = xla_dequantize_inputs()

import numpy as np
import copy

def xla_dynamic_slice_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.arange(10, dtype=np.int32)
    start_indices = np.array([2], dtype=np.int32)
    size_indices = np.array([5], dtype=np.int32)
    name = "slice_1d_basic"
    list_of_inputs.append(copy.deepcopy({
        "input": input_tensor,
        "start_indices": start_indices,
        "size_indices": size_indices,
        "name": name
    }))

    # Input 2
    input_tensor = np.arange(16, dtype=np.float32).reshape(4, 4)
    start_indices = np.array([1, 1], dtype=np.int32)
    size_indices = np.array([2, 2], dtype=np.int32)
    name = "slice_2d_center"
    list_of_inputs.append(copy.deepcopy({
        "input": input_tensor,
        "start_indices": start_indices,
        "size_indices": size_indices,
        "name": name
    }))

    # Input 3
    input_tensor = np.arange(27, dtype=np.int64).reshape(3, 3, 3)
    start_indices = np.array([0, 1, 1], dtype=np.int32)
    size_indices = np.array([2, 2, 2], dtype=np.int32)
    name = "slice_3d"
    list_of_inputs.append(copy.deepcopy({
        "input": input_tensor,
        "start_indices": start_indices,
        "size_indices": size_indices,
        "name": name
    }))

    # Input 4
    input_tensor = np.array([[-1, -2, -3], [-4, -5, -6]], dtype=np.int32)
    start_indices = np.array([0, 0], dtype=np.int32)
    size_indices = np.array([1, 3], dtype=np.int32)
    name = "slice_negative_values"
    list_of_inputs.append(copy.deepcopy({
        "input": input_tensor,
        "start_indices": start_indices,
        "size_indices": size_indices,
        "name": name
    }))

    # Input 5
    input_tensor = np.random.randn(5, 5).astype(np.float64)
    start_indices = np.array([2, 2], dtype=np.int32)
    size_indices = np.array([3, 3], dtype=np.int32)
    name = "slice_random_float"
    list_of_inputs.append(copy.deepcopy({
        "input": input_tensor,
        "start_indices": start_indices,
        "size_indices": size_indices,
        "name": name
    }))

    # Input 6
    input_tensor = np.arange(8, dtype=np.int32).reshape(2, 2, 2)
    start_indices = np.array([1, 0, 0], dtype=np.int32)
    size_indices = np.array([1, 2, 2], dtype=np.int32)
    name = "slice_edge_3d"
    list_of_inputs.append(copy.deepcopy({
        "input": input_tensor,
        "start_indices": start_indices,
        "size_indices": size_indices,
        "name": name
    }))

    # Input 7
    input_tensor = np.array([[1]], dtype=np.int32)
    start_indices = np.array([0, 0], dtype=np.int32)
    size_indices = np.array([1, 1], dtype=np.int32)
    name = "slice_single_element"
    list_of_inputs.append(copy.deepcopy({
        "input": input_tensor,
        "start_indices": start_indices,
        "size_indices": size_indices,
        "name": name
    }))

    # Input 8
    input_tensor = np.arange(20, dtype=np.int32)
    start_indices = np.array([0], dtype=np.int32)
    size_indices = np.array([10], dtype=np.int32)
    name = "slice_start_zero"
    list_of_inputs.append(copy.deepcopy({
        "input": input_tensor,
        "start_indices": start_indices,
        "size_indices": size_indices,
        "name": name
    }))

    # Input 9
    input_tensor = np.arange(60, dtype=np.int32).reshape(3, 4, 5)
    start_indices = np.array([1, 2, 3], dtype=np.int32)
    size_indices = np.array([2, 2, 2], dtype=np.int32)
    name = "slice_high_dim"
    list_of_inputs.append(copy.deepcopy({
        "input": input_tensor,
        "start_indices": start_indices,
        "size_indices": size_indices,
        "name": name
    }))

    # Input 10
    input_tensor = np.linspace(0, 1, 12, dtype=np.float32).reshape(3, 4)
    start_indices = np.array([0, 2], dtype=np.int32)
    size_indices = np.array([3, 2], dtype=np.int32)
    name = "slice_float_linspace"
    list_of_inputs.append(copy.deepcopy({
        "input": input_tensor,
        "start_indices": start_indices,
        "size_indices": size_indices,
        "name": name
    }))

    return list_of_inputs

generated_inputs["xla.dynamic_slice"] = xla_dynamic_slice_inputs()

import numpy as np
import copy

def xla_dynamic_update_slice_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = np.zeros((5,), dtype=np.float32)
    update = np.array([1.0, 2.0], dtype=np.float32)
    indices = np.array([1], dtype=np.int32)
    name = "1d_basic"
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr,
        "update": update,
        "indices": indices,
        "name": name
    }))

    # Input 2
    input_arr = np.arange(9, dtype=np.int32).reshape(3, 3)
    update = np.array([[99]], dtype=np.int32)
    indices = np.array([1, 1], dtype=np.int32)
    name = "2d_center"
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr,
        "update": update,
        "indices": indices,
        "name": name
    }))

    # Input 3
    input_arr = np.ones((4, 4), dtype=np.float64)
    update = np.array([[5.5, 6.6]], dtype=np.float64)
    indices = np.array([2, 1], dtype=np.int32)
    name = "2d_row_update"
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr,
        "update": update,
        "indices": indices,
        "name": name
    }))

    # Input 4
    input_arr = np.zeros((2, 3, 4), dtype=np.int32)
    update = np.ones((1, 2, 2), dtype=np.int32) * -3
    indices = np.array([1, 1, 1], dtype=np.int32)
    name = "3d_negative_update"
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr,
        "update": update,
        "indices": indices,
        "name": name
    }))

    # Input 5
    input_arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    update = np.array([[[100.0]]], dtype=np.float32)
    indices = np.array([0, 2, 3], dtype=np.int32)
    name = "3d_single_point"
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr,
        "update": update,
        "indices": indices,
        "name": name
    }))

    # Input 6
    input_arr = np.zeros((6,), dtype=np.int64)
    update = np.array([7, 8, 9], dtype=np.int64)
    indices = np.array([2], dtype=np.int32)
    name = "1d_longer_update"
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr,
        "update": update,
        "indices": indices,
        "name": name
    }))

    # Input 7
    input_arr = np.ones((3, 3, 3), dtype=np.float32)
    update = np.zeros((2, 2, 2), dtype=np.float32)
    indices = np.array([1, 0, 1], dtype=np.int32)
    name = "3d_block_update"
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr,
        "update": update,
        "indices": indices,
        "name": name
    }))

    # Input 8
    input_arr = np.full((4, 4), -1, dtype=np.int32)
    update = np.array([[2, 2], [2, 2]], dtype=np.int32)
    indices = np.array([0, 0], dtype=np.int32)
    name = "2d_top_left"
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr,
        "update": update,
        "indices": indices,
        "name": name
    }))

    # Input 9
    input_arr = np.arange(16, dtype=np.float64).reshape(4, 4)
    update = np.array([[50.5]], dtype=np.float64)
    indices = np.array([3, 3], dtype=np.int32)
    name = "2d_bottom_right"
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr,
        "update": update,
        "indices": indices,
        "name": name
    }))

    # Input 10
    input_arr = np.zeros((2, 2, 2, 2), dtype=np.int32)
    update = np.ones((1, 1, 1, 1), dtype=np.int32) * 42
    indices = np.array([1, 1, 1, 1], dtype=np.int32)
    name = "4d_single_update"
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr,
        "update": update,
        "indices": indices,
        "name": name
    }))

    return list_of_inputs

generated_inputs["xla.dynamic_update_slice"] = xla_dynamic_update_slice_inputs()

import numpy as np
import copy

def xla_key_value_sort_inputs():
    list_of_inputs = []

    keys = np.array([3, 1, 2], dtype=np.int32)
    values = np.array([30, 10, 20], dtype=np.int32)
    name = "simple_1d_int"
    list_of_inputs.append(copy.deepcopy({"keys": keys, "values": values, "name": name}))

    keys = np.array([0.2, -1.5, 3.3, 2.2], dtype=np.float32)
    values = np.array([2, 4, 1, 3], dtype=np.int32)
    name = "float_keys"
    list_of_inputs.append(copy.deepcopy({"keys": keys, "values": values, "name": name}))

    keys = np.array([[3, 1], [2, 4]], dtype=np.int32)
    values = np.array([[30, 10], [20, 40]], dtype=np.int32)
    name = "2d_int"
    list_of_inputs.append(copy.deepcopy({"keys": keys, "values": values, "name": name}))

    keys = np.array([[-1, -3, -2]], dtype=np.int32)
    values = np.array([[10, 30, 20]], dtype=np.int32)
    name = "negative_keys"
    list_of_inputs.append(copy.deepcopy({"keys": keys, "values": values, "name": name}))

    keys = np.array([5], dtype=np.int64)
    values = np.array([50], dtype=np.int64)
    name = "single_element"
    list_of_inputs.append(copy.deepcopy({"keys": keys, "values": values, "name": name}))

    keys = np.array([[1.1, 2.2], [3.3, 0.0]], dtype=np.float64)
    values = np.array([[11, 22], [33, 0]], dtype=np.int32)
    name = "float64_keys"
    list_of_inputs.append(copy.deepcopy({"keys": keys, "values": values, "name": name}))

    keys = np.array([3, 3, 1, 2], dtype=np.int32)
    values = np.array([300, 301, 100, 200], dtype=np.int32)
    name = "duplicate_keys"
    list_of_inputs.append(copy.deepcopy({"keys": keys, "values": values, "name": name}))

    keys = np.array([[9, 7, 8], [6, 5, 4]], dtype=np.int32)
    values = np.array([[90, 70, 80], [60, 50, 40]], dtype=np.int32)
    name = "2d_rectangular"
    list_of_inputs.append(copy.deepcopy({"keys": keys, "values": values, "name": name}))

    keys = np.array([0, -1, 1, 0], dtype=np.int32)
    values = np.array([0, -10, 10, 5], dtype=np.int32)
    name = "zeros_and_negatives"
    list_of_inputs.append(copy.deepcopy({"keys": keys, "values": values, "name": name}))

    keys = np.array([1.5, 1.5, 1.5], dtype=np.float32)
    values = np.array([15, 16, 17], dtype=np.int32)
    name = "all_equal_keys"
    list_of_inputs.append(copy.deepcopy({"keys": keys, "values": values, "name": name}))

    keys = np.array([[[3, 1], [2, 4]]], dtype=np.int32)
    values = np.array([[[30, 10], [20, 40]]], dtype=np.int32)
    name = "3d_tensor"
    list_of_inputs.append(copy.deepcopy({"keys": keys, "values": values, "name": name}))

    return list_of_inputs

generated_inputs["xla.key_value_sort"] = xla_key_value_sort_inputs()

import numpy as np
import copy

def xla_pad_inputs():
    list_of_inputs = []

    input_dict = {
        "input": np.array([1, 2, 3], dtype=np.int32),
        "padding_value": np.array(0, dtype=np.int32),
        "padding_low": np.array([1], dtype=np.int32),
        "padding_high": np.array([2], dtype=np.int32),
        "padding_interior": np.array([0], dtype=np.int32),
        "name": "pad_1d_basic"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.array([[1, 2], [3, 4]], dtype=np.float32),
        "padding_value": np.array(1.5, dtype=np.float32),
        "padding_low": np.array([1, 1], dtype=np.int32),
        "padding_high": np.array([1, 2], dtype=np.int32),
        "padding_interior": np.array([0, 0], dtype=np.int32),
        "name": "pad_2d_float"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.array([[[-1, -2], [-3, -4]]], dtype=np.int64),
        "padding_value": np.array(-9, dtype=np.int64),
        "padding_low": np.array([0, 1, 1], dtype=np.int32),
        "padding_high": np.array([1, 0, 2], dtype=np.int32),
        "padding_interior": np.array([0, 0, 0], dtype=np.int32),
        "name": "pad_3d_negative"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.array([10.5, -2.3, 7.7], dtype=np.float64),
        "padding_value": np.array(3.14, dtype=np.float64),
        "padding_low": np.array([2], dtype=np.int32),
        "padding_high": np.array([2], dtype=np.int32),
        "padding_interior": np.array([1], dtype=np.int32),
        "name": "pad_1d_interior"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.array([[1]], dtype=np.int32),
        "padding_value": np.array(5, dtype=np.int32),
        "padding_low": np.array([2, 2], dtype=np.int32),
        "padding_high": np.array([2, 2], dtype=np.int32),
        "padding_interior": np.array([1, 1], dtype=np.int32),
        "name": "pad_single_element"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
        "padding_value": np.array(-1, dtype=np.int32),
        "padding_low": np.array([0, 1], dtype=np.int32),
        "padding_high": np.array([1, 0], dtype=np.int32),
        "padding_interior": np.array([0, 2], dtype=np.int32),
        "name": "pad_2d_mixed"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.array([[[1.1], [2.2]], [[3.3], [4.4]]], dtype=np.float32),
        "padding_value": np.array(0.0, dtype=np.float32),
        "padding_low": np.array([1, 0, 1], dtype=np.int32),
        "padding_high": np.array([0, 1, 1], dtype=np.int32),
        "padding_interior": np.array([1, 0, 0], dtype=np.int32),
        "name": "pad_3d_float"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.array([5, 6, 7, 8], dtype=np.int64),
        "padding_value": np.array(99, dtype=np.int64),
        "padding_low": np.array([0], dtype=np.int32),
        "padding_high": np.array([3], dtype=np.int32),
        "padding_interior": np.array([2], dtype=np.int32),
        "name": "pad_1d_large_high"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.array([[[-5]]], dtype=np.int32),
        "padding_value": np.array(7, dtype=np.int32),
        "padding_low": np.array([1, 1, 1], dtype=np.int32),
        "padding_high": np.array([1, 1, 1], dtype=np.int32),
        "padding_interior": np.array([0, 0, 0], dtype=np.int32),
        "name": "pad_cube"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.array([[1.5, 2.5]], dtype=np.float64),
        "padding_value": np.array(-3.5, dtype=np.float64),
        "padding_low": np.array([1, 0], dtype=np.int32),
        "padding_high": np.array([0, 1], dtype=np.int32),
        "padding_interior": np.array([1, 1], dtype=np.int32),
        "name": "pad_float64_edge"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["xla.pad"] = xla_pad_inputs()

import numpy as np
import copy

def xla_reduce_precision_inputs():
    list_of_inputs = []

    # Input 1
    operand = np.array([1.5, -2.3, 3.7], dtype=np.float32)
    exponent_bits = 5
    mantissa_bits = 10
    list_of_inputs.append(copy.deepcopy({
        "operand": operand,
        "exponent_bits": exponent_bits,
        "mantissa_bits": mantissa_bits
    }))

    # Input 2
    operand = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    exponent_bits = 8
    mantissa_bits = 23
    list_of_inputs.append(copy.deepcopy({
        "operand": operand,
        "exponent_bits": exponent_bits,
        "mantissa_bits": mantissa_bits
    }))

    # Input 3
    operand = np.array([-1.25, -0.5, 0.0, 0.5, 1.25], dtype=np.float32)
    exponent_bits = 4
    mantissa_bits = 7
    list_of_inputs.append(copy.deepcopy({
        "operand": operand,
        "exponent_bits": exponent_bits,
        "mantissa_bits": mantissa_bits
    }))

    # Input 4
    operand = np.random.randn(3, 3).astype(np.float32)
    exponent_bits = 6
    mantissa_bits = 9
    list_of_inputs.append(copy.deepcopy({
        "operand": operand,
        "exponent_bits": exponent_bits,
        "mantissa_bits": mantissa_bits
    }))

    # Input 5
    operand = np.random.uniform(-10, 10, (2, 4, 3)).astype(np.float64)
    exponent_bits = 10
    mantissa_bits = 20
    list_of_inputs.append(copy.deepcopy({
        "operand": operand,
        "exponent_bits": exponent_bits,
        "mantissa_bits": mantissa_bits
    }))

    # Input 6
    operand = np.array([[[-1.1], [2.2]], [[-3.3], [4.4]]], dtype=np.float32)
    exponent_bits = 3
    mantissa_bits = 5
    list_of_inputs.append(copy.deepcopy({
        "operand": operand,
        "exponent_bits": exponent_bits,
        "mantissa_bits": mantissa_bits
    }))

    # Input 7
    operand = np.linspace(-5, 5, 10, dtype=np.float32)
    exponent_bits = 7
    mantissa_bits = 12
    list_of_inputs.append(copy.deepcopy({
        "operand": operand,
        "exponent_bits": exponent_bits,
        "mantissa_bits": mantissa_bits
    }))

    # Input 8
    operand = np.random.randn(1).astype(np.float64)
    exponent_bits = 11
    mantissa_bits = 52
    list_of_inputs.append(copy.deepcopy({
        "operand": operand,
        "exponent_bits": exponent_bits,
        "mantissa_bits": mantissa_bits
    }))

    # Input 9
    operand = np.zeros((5, 5), dtype=np.float32)
    exponent_bits = 2
    mantissa_bits = 3
    list_of_inputs.append(copy.deepcopy({
        "operand": operand,
        "exponent_bits": exponent_bits,
        "mantissa_bits": mantissa_bits
    }))

    # Input 10
    operand = np.full((2, 2, 2), 7.77, dtype=np.float64)
    exponent_bits = 9
    mantissa_bits = 15
    list_of_inputs.append(copy.deepcopy({
        "operand": operand,
        "exponent_bits": exponent_bits,
        "mantissa_bits": mantissa_bits
    }))

    # Input 11
    operand = np.array([1e-10, 1e10, -1e5], dtype=np.float32)
    exponent_bits = 5
    mantissa_bits = 8
    list_of_inputs.append(copy.deepcopy({
        "operand": operand,
        "exponent_bits": exponent_bits,
        "mantissa_bits": mantissa_bits
    }))

    return list_of_inputs

generated_inputs["xla.reduce_precision"] = xla_reduce_precision_inputs()

import numpy as np
import copy

def xla_rng_bit_generator_inputs():
    list_of_inputs = []

    input_dict = {
        "algorithm": "philox",
        "initial_state": "seed_123",
        "shape": [2, 3],
        "dtype": np.uint32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "algorithm": "threefry",
        "initial_state": "state_abc",
        "shape": [4],
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "algorithm": "default",
        "initial_state": "init_state_001",
        "shape": [1, 2, 3],
        "dtype": np.uint64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "algorithm": "philox",
        "initial_state": "negative_seed_-1",
        "shape": [5, 5],
        "dtype": np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "algorithm": "threefry",
        "initial_state": "long_seed_value_999999",
        "shape": [2, 2, 2, 2],
        "dtype": np.uint32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "algorithm": "default",
        "initial_state": "zero_state",
        "shape": [10],
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "algorithm": "philox",
        "initial_state": "state_xyz",
        "shape": [3, 1],
        "dtype": np.uint64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "algorithm": "threefry",
        "initial_state": "state_with_special_chars_!@#",
        "shape": [6, 0],
        "dtype": np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "algorithm": "default",
        "initial_state": "empty_like_state",
        "shape": [],
        "dtype": np.uint32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "algorithm": "philox",
        "initial_state": "very_large_shape",
        "shape": [2, 3, 4, 5],
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["xla.rng_bit_generator"] = xla_rng_bit_generator_inputs()

import numpy as np
import copy

def xla_sort_inputs():
    list_of_inputs = []

    input_dict = {
        "input": np.array([3, 1, 2, 5, 4], dtype=np.int32),
        "name": "simple_int_sort"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.array([-1, -3, -2, -5, -4], dtype=np.int32),
        "name": "negative_int_sort"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.array([1.5, 3.2, 0.7, 2.8], dtype=np.float32),
        "name": "float_sort"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.array([[3, 2, 1], [6, 5, 4]], dtype=np.int64),
        "name": "2d_int_sort"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.array([[1.1, 3.3], [2.2, 0.0]], dtype=np.float64),
        "name": "2d_float_sort"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.array([[[3, 1], [2, 4]], [[6, 5], [8, 7]]], dtype=np.int32),
        "name": "3d_tensor_sort"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.array([10], dtype=np.int32),
        "name": "single_element"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.array([], dtype=np.float32),
        "name": "empty_tensor"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.array([5, 5, 5, 5], dtype=np.int32),
        "name": "duplicate_values"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.array([100, -100, 50, -50, 0], dtype=np.int64),
        "name": "mixed_values"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.array([[[-1.1, 2.2], [3.3, -4.4]]], dtype=np.float32),
        "name": "3d_float_mixed"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["xla.sort"] = xla_sort_inputs()