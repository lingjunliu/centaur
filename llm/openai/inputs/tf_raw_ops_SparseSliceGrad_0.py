
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_SparseSliceGrad_inputs():
    def add_case(name, backprop_vals, backprop_dtype, start_list, output_indices_list, extra_inputs=0):
        start = np.array(start_list, dtype=np.int64)
        output_indices = np.array(output_indices_list, dtype=np.int64)
        rank = output_indices.shape[1]
        keys_o = tuple(output_indices[:, i] for i in reversed(range(rank)))
        order_o = np.lexsort(keys_o)
        output_indices = output_indices[order_o]
        backprop_val_grad = np.array(backprop_vals, dtype=backprop_dtype)[order_o]

        I_sel = output_indices + start
        if extra_inputs > 0:
            base = I_sel.max(axis=0) + 5
            extras = []
            for e in range(extra_inputs):
                extras.append(base + e + np.arange(rank))
            input_indices = np.vstack([I_sel, np.array(extras, dtype=np.int64)])
        else:
            input_indices = I_sel

        input_indices = np.array(input_indices, dtype=np.int64)
        keys_i = tuple(input_indices[:, i] for i in reversed(range(rank)))
        order_i = np.lexsort(keys_i)
        input_indices = input_indices[order_i]

        input_dict = {
            "name": name,
            "backprop_val_grad": backprop_val_grad,
            "input_indices": input_indices,
            "input_start": start,
            "output_indices": output_indices
        }
        return input_dict

    list_of_inputs = []

    list_of_inputs.append(copy.deepcopy(add_case(
        name="sparse_slice_grad_case_1_float32_rank1",
        backprop_vals=[1.0, -2.5],
        backprop_dtype=np.float32,
        start_list=[2],
        output_indices_list=[[0], [3]],
        extra_inputs=1
    )))

    list_of_inputs.append(copy.deepcopy(add_case(
        name="sparse_slice_grad_case_2_int32_rank2",
        backprop_vals=[5, -1, 0],
        backprop_dtype=np.int32,
        start_list=[1, 2],
        output_indices_list=[[0, 1], [1, 2], [2, 3]],
        extra_inputs=2
    )))

    list_of_inputs.append(copy.deepcopy(add_case(
        name="sparse_slice_grad_case_3_float64_rank3",
        backprop_vals=[-0.5, 2.0, -3.5, 4.75],
        backprop_dtype=np.float64,
        start_list=[0, 0, 1],
        output_indices_list=[[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]],
        extra_inputs=2
    )))

    list_of_inputs.append(copy.deepcopy(add_case(
        name="sparse_slice_grad_case_4_uint8_rank2",
        backprop_vals=[255],
        backprop_dtype=np.uint8,
        start_list=[5, 0],
        output_indices_list=[[2, 2]],
        extra_inputs=3
    )))

    list_of_inputs.append(copy.deepcopy(add_case(
        name="sparse_slice_grad_case_5_int16_rank4",
        backprop_vals=[-1000, 0, 1000, -2000, 2000],
        backprop_dtype=np.int16,
        start_list=[1, 1, 1, 1],
        output_indices_list=[[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]],
        extra_inputs=1
    )))

    list_of_inputs.append(copy.deepcopy(add_case(
        name="sparse_slice_grad_case_6_int8_rank3",
        backprop_vals=[-128, 127],
        backprop_dtype=np.int8,
        start_list=[3, 0, 2],
        output_indices_list=[[0, 0, 0], [2, 1, 0]],
        extra_inputs=1
    )))

    list_of_inputs.append(copy.deepcopy(add_case(
        name="sparse_slice_grad_case_7_complex64_rank2",
        backprop_vals=[1+2j, -3+0.5j, -1j],
        backprop_dtype=np.complex64,
        start_list=[0, 4],
        output_indices_list=[[1, 0], [2, 1], [3, 2]],
        extra_inputs=0
    )))

    list_of_inputs.append(copy.deepcopy(add_case(
        name="sparse_slice_grad_case_8_int64_rank1",
        backprop_vals=[0, -10, 20, -30],
        backprop_dtype=np.int64,
        start_list=[0],
        output_indices_list=[[0], [1], [2], [4]],
        extra_inputs=2
    )))

    list_of_inputs.append(copy.deepcopy(add_case(
        name="sparse_slice_grad_case_9_float32_rank2_sorted",
        backprop_vals=[0.1, -0.2, 0.3],
        backprop_dtype=np.float32,
        start_list=[2, 3],
        output_indices_list=[[0, 0], [0, 1], [1, 0]],
        extra_inputs=0
    )))

    list_of_inputs.append(copy.deepcopy(add_case(
        name="sparse_slice_grad_case_10_complex128_rank1",
        backprop_vals=[3.5-2.5j],
        backprop_dtype=np.complex128,
        start_list=[10],
        output_indices_list=[[5]],
        extra_inputs=2
    )))

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseSliceGrad"] = tf_raw_ops_SparseSliceGrad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SparseSliceGrad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseSliceGrad'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.SparseSliceGrad', generated_inputs['tf.raw_ops.SparseSliceGrad'], lib="tf", suffix=0)
