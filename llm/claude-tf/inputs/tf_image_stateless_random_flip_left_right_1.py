
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_image_stateless_random_flip_left_right_inputs():
    list_of_inputs = []
    
    image = np.array([[[1], [2]], [[3], [4]]], dtype=np.float32)
    seed = (2, 3)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.array([[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]], dtype=np.float32)
    seed = (0, 1)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(10, 10, 3).astype(np.float32)
    seed = (42, 100)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(5, 20, 20, 3).astype(np.float32)
    seed = (123, 456)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(64, 64, 1).astype(np.float32)
    seed = (7, 13)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(2, 32, 32, 4).astype(np.float32)
    seed = (99, 88)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.randn(15, 15, 3).astype(np.float32)
    seed = (1000, 2000)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.randint(0, 256, size=(3, 28, 28, 1), dtype=np.uint8)
    seed = (50, 75)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.randint(-100, 100, size=(8, 8, 3), dtype=np.int32)
    seed = (0, 0)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(1, 5, 5, 10).astype(np.float32)
    seed = (999, 111)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.image.stateless_random_flip_left_right_1"] = tf_image_stateless_random_flip_left_right_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.stateless_random_flip_left_right_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.stateless_random_flip_left_right_1'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.stateless_random_flip_left_right', generated_inputs['tf.image.stateless_random_flip_left_right_1'], lib="tf", suffix=1)
