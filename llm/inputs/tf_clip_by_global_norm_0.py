
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_clip_by_global_norm_inputs():
    list_of_inputs = []

    # Input 1
    t_list = [tf.constant([1.0, 2.0]), tf.constant([3.0, 4.0])]
    clip_norm = tf.constant(5.0)
    use_norm = tf.constant(6.0)
    name = "clip1"
    input_dict = {"t_list": t_list, "clip_norm": clip_norm, "use_norm": use_norm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    t_list = [tf.constant([1.0, 2.0, 3.0]), tf.constant([4.0, 5.0])]
    clip_norm = tf.constant(3.0)
    use_norm = tf.constant(2.0)
    name = "clip2"
    input_dict = {"t_list": t_list, "clip_norm": clip_norm, "use_norm": use_norm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    t_list = [tf.constant([[1.0, 2.0], [3.0, 4.0]]), tf.constant([5.0, 6.0])]
    clip_norm = tf.constant(7.0)
    use_norm = tf.constant(8.0)
    name = "clip3"
    input_dict = {"t_list": t_list, "clip_norm": clip_norm, "use_norm": use_norm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    t_list = [tf.constant([1.0]), tf.constant([2.0])]
    clip_norm = tf.constant(1.0)
    use_norm = tf.constant(0.5)
    name = "clip4"
    input_dict = {"t_list": t_list, "clip_norm": clip_norm, "use_norm": use_norm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    t_list = [tf.constant([1.0, 2.0]), tf.constant([3.0, 4.0])]
    clip_norm = tf.constant(10.0)
    use_norm = tf.constant(9.0)
    name = "clip5"
    input_dict = {"t_list": t_list, "clip_norm": clip_norm, "use_norm": use_norm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    t_list = [tf.constant([1.0, 2.0]), tf.constant([3.0, 4.0])]
    clip_norm = tf.constant(4.0)
    use_norm = tf.constant(5.0)
    name = "clip6"
    input_dict = {"t_list": t_list, "clip_norm": clip_norm, "use_norm": use_norm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    t_list = [tf.constant([-1.0, -2.0]), tf.constant([-3.0, -4.0])]
    clip_norm = tf.constant(5.0)
    use_norm = tf.constant(6.0)
    name = "clip7"
    input_dict = {"t_list": t_list, "clip_norm": clip_norm, "use_norm": use_norm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    t_list = [tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), tf.constant([7.0, 8.0, 9.0])]
    clip_norm = tf.constant(12.0)
    use_norm = tf.constant(11.0)
    name = "clip8"
    input_dict = {"t_list": t_list, "clip_norm": clip_norm, "use_norm": use_norm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    t_list = [tf.constant(np.array([1.0, 2.0, 3.0]).astype(np.float32)), tf.constant(np.array([4.0, 5.0]).astype(np.float32))]
    clip_norm = tf.constant(np.array(3.0).astype(np.float32))
    use_norm = tf.constant(np.array(2.0).astype(np.float32))
    name = "clip9"
    input_dict = {"t_list": t_list, "clip_norm": clip_norm, "use_norm": use_norm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    t_list = [tf.constant(np.random.rand(2,3,4).astype(np.float32)), tf.constant(np.random.rand(5,2).astype(np.float32))]
    clip_norm = tf.constant(np.array(1.5).astype(np.float32))
    use_norm = tf.constant(np.array(0.7).astype(np.float32))
    name = "clip10"
    input_dict = {"t_list": t_list, "clip_norm": clip_norm, "use_norm": use_norm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
input_list = tf_clip_by_global_norm_inputs()
generated_inputs["tf.clip_by_global_norm"] = input_list

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.clip_by_global_norm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.clip_by_global_norm'.")

check_valid('tf.clip_by_global_norm', generated_inputs['tf.clip_by_global_norm'], lib="tf", suffix=0)
