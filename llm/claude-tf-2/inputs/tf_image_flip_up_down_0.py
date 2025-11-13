
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_image_flip_up_down_inputs():
    list_of_inputs = []
    
    image = np.array([[[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]],
                      [[7.0, 8.0, 9.0],
                       [10.0, 11.0, 12.0]]], dtype=np.float32)
    input_dict = {"image": image}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(2, 4, 4, 3).astype(np.float32)
    input_dict = {"image": image}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(5, 5, 1).astype(np.float32)
    input_dict = {"image": image}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(3, 3, 4).astype(np.float32)
    input_dict = {"image": image}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(1, 6, 8, 3).astype(np.float32)
    input_dict = {"image": image}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.array([[[-1.0, -2.0, -3.0],
                       [-4.0, -5.0, -6.0]],
                      [[7.0, 8.0, 9.0],
                       [10.0, 11.0, 12.0]]], dtype=np.float32)
    input_dict = {"image": image}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(5, 10, 10, 3).astype(np.float32)
    input_dict = {"image": image}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.randint(0, 256, size=(4, 4, 3), dtype=np.uint8)
    input_dict = {"image": image}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)
    input_dict = {"image": image}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.array([[[[1.0, -2.0, 3.0],
                        [-4.0, 5.0, -6.0]],
                       [[7.0, -8.0, 9.0],
                        [-10.0, 11.0, -12.0]]]], dtype=np.float32)
    input_dict = {"image": image}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.image.flip_up_down"] = tf_image_flip_up_down_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.flip_up_down' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.flip_up_down'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.flip_up_down', generated_inputs['tf.image.flip_up_down'], lib="tf", suffix=0)
