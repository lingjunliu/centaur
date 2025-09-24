
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_top_k_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([1, 2, 98, 1, 1, 99, 3, 1, 3, 96, 4, 1], dtype=np.int32)
    k = 3
    sorted_bool = True
    index_type = tf.int32
    name_str = "top_3"
    input_dict = {"input": input_tensor, "k": k, "sorted": sorted_bool, "index_type": index_type, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.normal(size=(3, 4, 5, 6)).astype(np.float32)
    k = 2
    sorted_bool = False
    index_type = tf.int64
    name_str = None
    input_dict = {"input": input_tensor, "k": k, "sorted": sorted_bool, "index_type": index_type, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.int32)
    k = 3
    sorted_bool = True
    index_type = tf.int16
    name_str = "topk_indices"
    input_dict = {"input": input_tensor, "k": k, "sorted": sorted_bool, "index_type": index_type, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: k=1, multi-dimensional input
    input_tensor = np.array([[1, 5, 2], [8, 3, 9]], dtype=np.int32)
    k = 1
    sorted_bool = True
    index_type = tf.int32
    name_str = None
    input_dict = {"input": input_tensor, "k": k, "sorted": sorted_bool, "index_type": index_type, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: negative values, sorted=False
    input_tensor = np.array([-1, -2, -3, -4, -5], dtype=np.int32)
    k = 3
    sorted_bool = False
    index_type = tf.int32
    name_str = None
    input_dict = {"input": input_tensor, "k": k, "sorted": sorted_bool, "index_type": index_type, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D Tensor
    input_tensor = np.random.rand(2, 3, 4).astype(np.float32)
    k = 2
    sorted_bool = True
    index_type = tf.int32
    name_str = "3d_topk"
    input_dict = {"input": input_tensor, "k": k, "sorted": sorted_bool, "index_type": index_type, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: k equals the size of the last dimension
    input_tensor = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    k = 5
    sorted_bool = True
    index_type = tf.int32
    name_str = None
    input_dict = {"input": input_tensor, "k": k, "sorted": sorted_bool, "index_type": index_type, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: all elements are the same
    input_tensor = np.array([1, 1, 1, 1, 1], dtype=np.int32)
    k = 3
    sorted_bool = True
    index_type = tf.int32
    name_str = None
    input_dict = {"input": input_tensor, "k": k, "sorted": sorted_bool, "index_type": index_type, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large values
    input_tensor = np.array([1000000, 2000000, 3000000, 4000000, 5000000], dtype=np.int32)
    k = 3
    sorted_bool = True
    index_type = tf.int32
    name_str = None
    input_dict = {"input": input_tensor, "k": k, "sorted": sorted_bool, "index_type": index_type, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Float input
    input_tensor = np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float32)
    k = 3
    sorted_bool = True
    index_type = tf.int32
    name_str = None
    input_dict = {"input": input_tensor, "k": k, "sorted": sorted_bool, "index_type": index_type, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: all negative values and unsorted
    input_tensor = np.array([-1.1, -2.2, -3.3, -4.4, -5.5], dtype=np.float32)
    k = 3
    sorted_bool = False
    index_type = tf.int32
    name_str = None
    input_dict = {"input": input_tensor, "k": k, "sorted": sorted_bool, "index_type": index_type, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.top_k"] = tf_math_top_k_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.top_k' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.top_k'.")

check_valid('tf.math.top_k', generated_inputs['tf.math.top_k'], lib="tf", suffix=0)
