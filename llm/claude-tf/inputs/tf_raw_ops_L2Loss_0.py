
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_l2loss_inputs():
    list_of_inputs = []
    
    t = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    name = "l2loss_1d"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    t = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    name = "l2loss_2d"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    t = np.array([[-1.0, -2.0], [3.0, -4.0]], dtype=np.float32)
    name = "l2loss_negative"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    t = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    name = "l2loss_3d"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    t = np.array([5.0], dtype=np.float32)
    name = "l2loss_scalar"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    t = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    name = "l2loss_float64"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    t = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    name = "l2loss_float16"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    t = np.zeros((3, 3), dtype=np.float32)
    name = "l2loss_zeros"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    t = np.ones((2, 2, 2, 2), dtype=np.float32)
    name = "l2loss_4d"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    t = np.array([1.0, -1.0, 2.0, -2.0], dtype=np.float32)
    name = "l2loss_mixed"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.L2Loss"] = tf_raw_ops_l2loss_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.L2Loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.L2Loss'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.L2Loss', generated_inputs['tf.raw_ops.L2Loss'], lib="tf", suffix=0)
