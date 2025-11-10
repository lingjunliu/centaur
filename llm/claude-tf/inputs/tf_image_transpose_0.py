
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_image_transpose_inputs():
    list_of_inputs = []
    
    image = np.array([[[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]],
                      [[7.0, 8.0, 9.0],
                       [10.0, 11.0, 12.0]]], dtype=np.float32)
    input_dict = {
        "image": image,
        "name": "transpose_op_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.array([[[1, 2], [3, 4]],
                      [[5, 6], [7, 8]],
                      [[9, 10], [11, 12]]], dtype=np.int32)
    input_dict = {
        "image": image,
        "name": "transpose_op_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(2, 3, 4, 3).astype(np.float32)
    input_dict = {
        "image": image,
        "name": "transpose_op_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(5, 7, 1).astype(np.float32)
    input_dict = {
        "image": image,
        "name": "transpose_op_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(1, 10, 8, 3).astype(np.float32)
    input_dict = {
        "image": image,
        "name": "transpose_op_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.array([[[-1.0, -2.0], [3.0, 4.0]],
                      [[-5.0, 6.0], [-7.0, 8.0]]], dtype=np.float32)
    input_dict = {
        "image": image,
        "name": "transpose_op_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(5, 4, 6, 3).astype(np.float32)
    input_dict = {
        "image": image,
        "name": "transpose_op_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(3, 3, 64).astype(np.float32)
    input_dict = {
        "image": image,
        "name": "transpose_op_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.randint(0, 256, size=(2, 6, 8, 3), dtype=np.uint8)
    input_dict = {
        "image": image,
        "name": "transpose_op_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.zeros((4, 5, 3), dtype=np.float32)
    input_dict = {
        "image": image,
        "name": "transpose_op_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.image.transpose"] = tf_image_transpose_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.transpose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.transpose'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.transpose', generated_inputs['tf.image.transpose'], lib="tf", suffix=0)
