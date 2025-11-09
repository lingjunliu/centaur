
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_image_stateless_random_flip_up_down_inputs():
    list_of_inputs = []
    
    image = np.array([[[1], [2]], [[3], [4]]], dtype=np.float32)
    seed = np.array([2, 3], dtype=np.int32)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.float32)
    seed = np.array([10, 20], dtype=np.int32)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.array([[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]], dtype=np.float32)
    seed = np.array([5, 7], dtype=np.int32)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(10, 8, 3).astype(np.float32)
    seed = np.array([100, 200], dtype=np.int32)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(4, 5, 5, 3).astype(np.float32)
    seed = np.array([42, 84], dtype=np.int32)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.array([[[-1, -2], [3, 4]], [[5, -6], [-7, 8]]], dtype=np.float32)
    seed = np.array([15, 25], dtype=np.int32)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(1, 6, 6, 1).astype(np.float32)
    seed = np.array([0, 1], dtype=np.int32)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    seed = np.array([999, 1000], dtype=np.int64)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(2, 4, 4, 4).astype(np.float32)
    seed = np.array([123, 456], dtype=np.int32)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.ones((3, 3, 1), dtype=np.float32)
    seed = np.array([50, 60], dtype=np.int32)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.image.stateless_random_flip_up_down_2"] = tf_image_stateless_random_flip_up_down_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.stateless_random_flip_up_down_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.stateless_random_flip_up_down_2'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.stateless_random_flip_up_down', generated_inputs['tf.image.stateless_random_flip_up_down_2'], lib="tf", suffix=2)
