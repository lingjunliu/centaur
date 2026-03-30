
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_clip_by_global_norm_inputs():
    list_of_inputs = []

    def convert_to_numpy(t_list):
        new_list = []
        for t in t_list:
            if t is not None and not isinstance(t, np.ndarray):
                new_list.append(np.array(t, dtype=np.float32))
            else:
                new_list.append(t)
        return new_list

    # Input 1
    t_list = [np.array([1.0, 2.0, 3.0], dtype=np.float32), np.array([4.0, 5.0, 6.0], dtype=np.float32)]
    clip_norm = np.array(5.0, dtype=np.float32)
    use_norm = np.array(7.0, dtype=np.float32)
    name = "clip_1"
    input_dict = {"t_list": t_list, "clip_norm": clip_norm, "use_norm": use_norm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    t_list = [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)]
    clip_norm = np.array(10.0, dtype=np.float32)
    use_norm = None
    name = "clip_2"
    input_dict = {"t_list": t_list, "clip_norm": clip_norm, "use_norm": use_norm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    t_list = [np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32), np.array([[5.0, -6.0], [-7.0, 8.0]], dtype=np.float32)]
    clip_norm = np.array(3.0, dtype=np.float32)
    use_norm = np.array(2.0, dtype=np.float32)
    name = "clip_3"
    input_dict = {"t_list": t_list, "clip_norm": clip_norm, "use_norm": use_norm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    t_list = [np.array([1.5, 2.5, 3.5], dtype=np.float32), np.array([4.5, 5.5, 6.5], dtype=np.float32)]
    clip_norm = np.array(6.0, dtype=np.float32)
    use_norm = None
    name = "clip_4"
    input_dict = {"t_list": t_list, "clip_norm": clip_norm, "use_norm": use_norm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    t_list = [np.array([0.1, 0.2, 0.3], dtype=np.float32), np.array([0.4, 0.5, 0.6], dtype=np.float32)]
    clip_norm = np.array(0.5, dtype=np.float32)
    use_norm = np.array(1.0, dtype=np.float32)
    name = "clip_5"
    input_dict = {"t_list": t_list, "clip_norm": clip_norm, "use_norm": use_norm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 (None in list)
    t_list = [np.array([1.0, 2.0], dtype=np.float32), None, np.array([3.0, 4.0], dtype=np.float32)]
    clip_norm = np.array(4.0, dtype=np.float32)
    use_norm = np.array(5.0, dtype=np.float32)
    name = "clip_7"
    input_dict = {"t_list": convert_to_numpy(t_list), "clip_norm": clip_norm, "use_norm": use_norm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 (different shapes)
    t_list = [np.array([1.0, 2.0], dtype=np.float32), np.array([[3.0], [4.0]], dtype=np.float32)]
    clip_norm = np.array(7.0, dtype=np.float32)
    use_norm = None
    name = "clip_8"
    input_dict = {"t_list": t_list, "clip_norm": clip_norm, "use_norm": use_norm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 (3D tensor)
    t_list = [np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)]
    clip_norm = np.array(12.0, dtype=np.float32)
    use_norm = np.array(11.0, dtype=np.float32)
    name = "clip_9"
    input_dict = {"t_list": t_list, "clip_norm": clip_norm, "use_norm": use_norm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 (clip_norm > norm)
    t_list = [np.array([1.0, 2.0, 3.0], dtype=np.float32), np.array([4.0, 5.0, 6.0], dtype=np.float32)]
    clip_norm = np.array(100.0, dtype=np.float32)
    use_norm = np.array(7.0, dtype=np.float32)
    name = "clip_10"
    input_dict = {"t_list": t_list, "clip_norm": clip_norm, "use_norm": use_norm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 - single tensor
    t_list = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]
    clip_norm = np.array(5.0, dtype=np.float32)
    use_norm = np.array(7.0, dtype=np.float32)
    name = "clip_11"
    input_dict = {"t_list": t_list, "clip_norm": clip_norm, "use_norm": use_norm, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.clip_by_global_norm"] = tf_clip_by_global_norm_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.clip_by_global_norm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.clip_by_global_norm'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.clip_by_global_norm', generated_inputs['tf.clip_by_global_norm'], lib="tf", suffix=0)
