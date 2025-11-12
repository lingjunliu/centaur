
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_image_random_flip_left_right_inputs():
    list_of_inputs = []
    
    image = np.array([[[1], [2]], [[3], [4]]], dtype=np.float32)
    seed = 5
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.array([[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]], dtype=np.float32)
    seed = 6
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(32, 32, 3).astype(np.float32)
    seed = 42
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(8, 64, 64, 3).astype(np.float32)
    seed = 100
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(128, 128, 4).astype(np.float32)
    seed = 0
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.ones((10, 10, 1), dtype=np.float32)
    seed = 999
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(1, 50, 50, 3).astype(np.float32)
    seed = 123
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(16, 16, 16).astype(np.float32)
    seed = 2023
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(4, 256, 256, 3).astype(np.float32)
    seed = 777
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(224, 224, 1).astype(np.float32)
    seed = 1
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.image.random_flip_left_right"] = tf_image_random_flip_left_right_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.random_flip_left_right' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.random_flip_left_right'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.random_flip_left_right', generated_inputs['tf.image.random_flip_left_right'], lib="tf", suffix=0)
