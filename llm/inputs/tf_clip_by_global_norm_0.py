
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_clip_by_global_norm_inputs():
    list_of_inputs = []

    # Input 1: Basic test case
    t_list = [tf.constant([1.0, 2.0, 3.0]), tf.constant([4.0, 5.0, 6.0])]
    clip_norm = tf.constant(5.0)
    use_norm = tf.constant(7.0)
    name = "clip_test_1"
    input_dict = {"t_list": t_list, "clip_norm": clip_norm, "use_norm": use_norm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different clip_norm
    t_list = [tf.constant([1.0, 2.0]), tf.constant([3.0, 4.0])]
    clip_norm = tf.constant(10.0)
    use_norm = tf.constant(4.0)
    name = "clip_test_2"
    input_dict = {"t_list": t_list, "clip_norm": clip_norm, "use_norm": use_norm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Zero tensors
    t_list = [tf.constant([0.0, 0.0]), tf.constant([0.0, 0.0])]
    clip_norm = tf.constant(1.0)
    use_norm = tf.constant(0.0)
    name = "clip_test_3"
    input_dict = {"t_list": t_list, "clip_norm": clip_norm, "use_norm": use_norm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Higher dimensional tensors
    t_list = [tf.constant([[1.0, 2.0], [3.0, 4.0]]), tf.constant([[5.0, 6.0], [7.0, 8.0]])]
    clip_norm = tf.constant(7.0)
    use_norm = tf.constant(10.0)
    name = "clip_test_5"
    input_dict = {"t_list": t_list, "clip_norm": clip_norm, "use_norm": use_norm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Scalar tensor
    t_list = [tf.constant(2.0), tf.constant(3.0)]
    clip_norm = tf.constant(4.0)
    use_norm = tf.constant(5.0)
    name = "clip_test_6"
    input_dict = {"t_list": t_list, "clip_norm": clip_norm, "use_norm": use_norm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))


    # Input 6: clip_norm close to 0
    t_list = [tf.constant([1.0, 2.0]), tf.constant([3.0, 4.0])]
    clip_norm = tf.constant(0.001)
    use_norm = tf.constant(1.0)
    name = "clip_test_8"
    input_dict = {"t_list": t_list, "clip_norm": clip_norm, "use_norm": use_norm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Mixed shapes - making the scalar a tensor of shape ()
    t_list = [tf.constant([1.0, 2.0]), tf.constant([3.0])]
    clip_norm = tf.constant(2.0)
    use_norm = tf.constant(3.0)
    name = "clip_test_9"
    input_dict = {"t_list": t_list, "clip_norm": clip_norm, "use_norm": use_norm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: different shapes - ensuring consistent rank
    t_list = [tf.constant([[1.0, 2.0], [3.0, 4.0]]), tf.constant([[5.0, 6.0], [7.0, 8.0]])]
    clip_norm = tf.constant(1.5)
    use_norm = tf.constant(2.0)
    name = "clip_test_11"
    input_dict = {"t_list": t_list, "clip_norm": clip_norm, "use_norm": use_norm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9:
    t_list = [tf.constant([[1.0, 2.0], [3.0, 4.0]]), tf.constant([[5.0,5.0]])]
    clip_norm = tf.constant(3.0)
    use_norm = tf.constant(4.0)
    name = "clip_test_12"
    input_dict = {"t_list": t_list, "clip_norm": clip_norm, "use_norm": use_norm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: all scalar tensors
    t_list = [tf.constant(1.0), tf.constant(2.0), tf.constant(3.0)]
    clip_norm = tf.constant(0.5)
    use_norm = tf.constant(1.0)
    name = "clip_test_13"
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
