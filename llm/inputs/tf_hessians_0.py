
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_hessians_inputs():
    list_of_inputs = []

    # Input 1: Basic case with single ys and xs
    ys = [tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))]
    xs = [tf.constant(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32))]
    gate_gradients = False
    aggregation_method = None
    name = "hessian_basic"
    input_dict = {"ys": ys, "xs": xs, "gate_gradients": gate_gradients, "aggregation_method": aggregation_method, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Multiple ys and xs
    ys = [tf.constant(np.array([1.0, 2.0], dtype=np.float32)), tf.constant(np.array([3.0, 4.0], dtype=np.float32))]
    xs = [tf.constant(np.array([5.0, 6.0], dtype=np.float32)), tf.constant(np.array([7.0, 8.0], dtype=np.float32))]
    gate_gradients = True
    aggregation_method = "SUM"
    name = "hessian_multiple"
    input_dict = {"ys": ys, "xs": xs, "gate_gradients": gate_gradients, "aggregation_method": aggregation_method, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: ys and xs with different shapes
    ys = [tf.constant(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32))]
    xs = [tf.constant(np.array([7.0, 8.0], dtype=np.float32))]
    gate_gradients = False
    aggregation_method = "MEAN"
    name = "hessian_diff_shape"
    input_dict = {"ys": ys, "xs": xs, "gate_gradients": gate_gradients, "aggregation_method": aggregation_method, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Rank 3 ys and xs
    ys = [tf.constant(np.random.rand(2, 3, 4).astype(np.float32))]
    xs = [tf.constant(np.random.rand(2, 3, 4).astype(np.float32))]
    gate_gradients = False
    aggregation_method = None
    name = "hessian_rank3"
    input_dict = {"ys": ys, "xs": xs, "gate_gradients": gate_gradients, "aggregation_method": aggregation_method, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Empty name
    ys = [tf.constant(np.array([1.0], dtype=np.float32))]
    xs = [tf.constant(np.array([2.0], dtype=np.float32))]
    gate_gradients = True
    aggregation_method = "SUM"
    name = ""
    input_dict = {"ys": ys, "xs": xs, "gate_gradients": gate_gradients, "aggregation_method": aggregation_method, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: ys and xs are the same tensor
    ys = [tf.constant(np.array([1.0, 2.0], dtype=np.float32))]
    xs = [ys[0]]
    gate_gradients = True
    aggregation_method = "EXPERIMENTAL_ACCUMULATE_N"
    name = "hessian_same_tensor"
    input_dict = {"ys": ys, "xs": xs, "gate_gradients": gate_gradients, "aggregation_method": aggregation_method, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7: Different aggregation method
    ys = [tf.constant(np.array([1.0, 2.0], dtype=np.float32))]
    xs = [tf.constant(np.array([3.0, 4.0], dtype=np.float32))]
    gate_gradients = False
    aggregation_method = "ADD_N"
    name = "hessian_add_n"
    input_dict = {"ys": ys, "xs": xs, "gate_gradients": gate_gradients, "aggregation_method": aggregation_method, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Rank 1 tensors
    ys = [tf.constant(np.array([1.0], dtype=np.float32))]
    xs = [tf.constant(np.array([2.0], dtype=np.float32))]
    gate_gradients = True
    aggregation_method = "SUM"
    name = "hessian_rank1"
    input_dict = {"ys": ys, "xs": xs, "gate_gradients": gate_gradients, "aggregation_method": aggregation_method, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D ys and 2D xs
    ys = [tf.constant(np.random.rand(2, 3, 4).astype(np.float32))]
    xs = [tf.constant(np.random.rand(2, 3).astype(np.float32))]
    gate_gradients = False
    aggregation_method = None
    name = "hessian_3d_ys_2d_xs"
    input_dict = {"ys": ys, "xs": xs, "gate_gradients": gate_gradients, "aggregation_method": aggregation_method, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Single element tensors
    ys = [tf.constant(np.array([1.0], dtype=np.float32))]
    xs = [tf.constant(np.array([2.0], dtype=np.float32))]
    gate_gradients = False
    aggregation_method = "MEAN"
    name = "hessian_single_element"
    input_dict = {"ys": ys, "xs": xs, "gate_gradients": gate_gradients, "aggregation_method": aggregation_method, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Scalar values
    ys = [tf.constant(np.array(1.0, dtype=np.float32))]
    xs = [tf.constant(np.array(2.0, dtype=np.float32))]
    gate_gradients = False
    aggregation_method = None
    name = "hessian_scalar"
    input_dict = {"ys": ys, "xs": xs, "gate_gradients": gate_gradients, "aggregation_method": aggregation_method, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: ys with multiple elements, xs a scalar
    ys = [tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32))]
    xs = [tf.constant(np.array(2.0, dtype=np.float32))]
    gate_gradients = True
    aggregation_method = "SUM"
    name = "hessian_ys_multiple_xs_scalar"
    input_dict = {"ys": ys, "xs": xs, "gate_gradients": gate_gradients, "aggregation_method": aggregation_method, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.hessians"] = tf_hessians_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.hessians' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.hessians'.")

check_valid('tf.hessians', generated_inputs['tf.hessians'], lib="tf", suffix=0)
