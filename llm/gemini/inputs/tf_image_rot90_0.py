
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_rot90_inputs():
    list_of_inputs = []

    # Input 1
    image = np.array([[[1],[2]],[[3],[4]]], dtype=np.int32)
    k = 1
    name = "rotate_90"
    input_dict = {"image": image, "k": k, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    image = np.array([[[1, 2, 3],[4, 5, 6]],[[7, 8, 9],[10, 11, 12]]], dtype=np.float32)
    k = 2
    name = None
    input_dict = {"image": image, "k": k, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    image = np.array([[[1, 2],[3, 4]],[[5, 6],[7, 8]]], dtype=np.int64)
    k = -1
    name = "rotate_minus_90"
    input_dict = {"image": image, "k": k, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    image = np.array([[[1],[2]],[[3],[4]],[[5],[6]]], dtype=np.float64)
    k = 0
    name = "rotate_0"
    input_dict = {"image": image, "k": k, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    image = np.array([[[1,1],[2,2]],[[3,3],[4,4]],[[5,5],[6,6]]], dtype=np.int32)
    k = 4
    name = "rotate_360"
    input_dict = {"image": image, "k": k, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    image = np.array([[[1, 2, 3, 4],[5, 6, 7, 8]],[[9, 10, 11, 12],[13, 14, 15, 16]]], dtype=np.float32)
    k = -2
    name = "rotate_minus_180"
    input_dict = {"image": image, "k": k, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    image = np.array([[[1,1,1],[2,2,2]],[[3,3,3],[4,4,4]],[[5,5,5],[6,6,6]]], dtype=np.int64)
    k = 10
    name = "rotate_large_k"
    input_dict = {"image": image, "k": k, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    image = np.array([[[1.0],[2.0]],[[3.0],[4.0]]], dtype=np.float32)
    k = -5
    name = "rotate_negative_large_k"
    input_dict = {"image": image, "k": k, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9, 4D Tensor
    image = np.random.rand(2, 3, 4, 3).astype(np.float32)
    k = 1
    name = "4d_tensor_rotate"
    input_dict = {"image": image, "k": k, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, 4D Tensor with negative k
    image = np.random.rand(1, 5, 5, 1).astype(np.int32)
    k = -3
    name = "4d_tensor_rotate_negative"
    input_dict = {"image": image, "k": k, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.rot90"] = tf_image_rot90_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.rot90' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.rot90'.")

check_valid('tf.image.rot90', generated_inputs['tf.image.rot90'], lib="tf", suffix=0)
