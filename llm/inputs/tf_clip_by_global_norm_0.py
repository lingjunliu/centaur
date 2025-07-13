
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_clip_by_global_norm_inputs():
    list_of_inputs = []

    # Input 1
    t_list = [tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32))]
    clip_norm = tf.constant(2.0, dtype=np.float32)
    use_norm = tf.constant(3.0, dtype=np.float32)
    name = "clip_norm_op_1"
    input_dict = {"t_list": t_list, "clip_norm": clip_norm, "use_norm": use_norm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    t_list = [tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))]
    clip_norm = tf.constant(5.0, dtype=np.float32)
    use_norm = tf.constant(4.0, dtype=np.float32)
    name = "clip_norm_op_2"
    input_dict = {"t_list": t_list, "clip_norm": clip_norm, "use_norm": use_norm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    t_list = [tf.constant(np.array([1.0, 2.0], dtype=np.float32)), tf.constant(np.array([3.0, 4.0], dtype=np.float32))]
    clip_norm = tf.constant(1.0, dtype=np.float32)
    use_norm = tf.constant(6.0, dtype=np.float32)
    name = "clip_norm_op_3"
    input_dict = {"t_list": t_list, "clip_norm": clip_norm, "use_norm": use_norm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    t_list = [tf.constant(np.array([-1.0, 2.0], dtype=np.float32)), tf.constant(np.array([3.0, -4.0], dtype=np.float32))]
    clip_norm = tf.constant(3.0, dtype=np.float32)
    use_norm = tf.constant(5.0, dtype=np.float32)
    name = "clip_norm_op_4"
    input_dict = {"t_list": t_list, "clip_norm": clip_norm, "use_norm": use_norm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    t_list = [tf.constant(np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32))]
    clip_norm = tf.constant(10.0, dtype=np.float32)
    use_norm = tf.constant(9.0, dtype=np.float32)
    name = "clip_norm_op_5"
    input_dict = {"t_list": t_list, "clip_norm": clip_norm, "use_norm": use_norm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    t_list = [tf.constant(np.array([1.0], dtype=np.float32))]
    clip_norm = tf.constant(0.5, dtype=np.float32)
    use_norm = tf.constant(1.0, dtype=np.float32)
    name = "clip_norm_op_6"
    input_dict = {"t_list": t_list, "clip_norm": clip_norm, "use_norm": use_norm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    t_list = [tf.constant(np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32))]
    clip_norm = tf.constant(5.0, dtype=np.float32)
    use_norm = tf.constant(1.0, dtype=np.float32)
    name = "clip_norm_op_7"
    input_dict = {"t_list": t_list, "clip_norm": clip_norm, "use_norm": use_norm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    t_list = [tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32)), tf.constant(np.array([[4.0, 5.0], [6.0, 7.0]], dtype=np.float32))]
    clip_norm = tf.constant(7.0, dtype=np.float32)
    use_norm = tf.constant(2.0, dtype=np.float32)
    name = "clip_norm_op_8"
    input_dict = {"t_list": t_list, "clip_norm": clip_norm, "use_norm": use_norm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    t_list = [tf.constant(np.array([-1.0, -2.0, -3.0], dtype=np.float32))]
    clip_norm = tf.constant(4.0, dtype=np.float32)
    use_norm = tf.constant(1.0, dtype=np.float32)
    name = "clip_norm_op_9"
    input_dict = {"t_list": t_list, "clip_norm": clip_norm, "use_norm": use_norm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    t_list = [tf.constant(np.array([[1.0, -2.0], [-3.0, 4.0]], dtype=np.float32))]
    clip_norm = tf.constant(6.0, dtype=np.float32)
    use_norm = tf.constant(8.0, dtype=np.float32)
    name = "clip_norm_op_10"
    input_dict = {"t_list": t_list, "clip_norm": clip_norm, "use_norm": use_norm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.clip_by_global_norm"] = tf_clip_by_global_norm_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.clip_by_global_norm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.clip_by_global_norm'.")

check_valid('tf.clip_by_global_norm', generated_inputs['tf.clip_by_global_norm'], lib="tf", suffix=0)
