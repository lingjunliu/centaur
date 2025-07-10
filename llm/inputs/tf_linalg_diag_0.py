
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_diag_inputs():
    list_of_inputs = []

    # Input 1
    diagonal = np.array([1, 2, 3])
    name = "diag1"
    k = 0
    num_rows = -1
    num_cols = -1
    padding_value = np.array(0)
    align = "RIGHT_LEFT"

    input_dict = {
        "diagonal": diagonal,
        "name": name,
        "k": k,
        "num_rows": num_rows,
        "num_cols": num_cols,
        "padding_value": padding_value,
        "align": align
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    diagonal = np.array([[1, 2], [3, 4]])
    name = "diag2"
    k = 1
    num_rows = -1
    num_cols = -1
    padding_value = np.array(0)
    align = "LEFT_RIGHT"
    input_dict = {
        "diagonal": diagonal,
        "name": name,
        "k": k,
        "num_rows": num_rows,
        "num_cols": num_cols,
        "padding_value": padding_value,
        "align": align
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    diagonal = np.array([[[8, 9, 0], [1, 2, 3], [0, 4, 5]], [[2, 3, 0], [6, 7, 9], [0, 9, 1]]])
    name = "diag3"
    k = (-1, 1)
    num_rows = 3
    num_cols = 3
    padding_value = np.array(9)
    align = "RIGHT_LEFT"
    input_dict = {
        "diagonal": diagonal,
        "name": name,
        "k": k,
        "num_rows": num_rows,
        "num_cols": num_cols,
        "padding_value": padding_value,
        "align": align
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    diagonal = np.array([1, 2])
    name = "diag4"
    k = -1
    num_rows = 3
    num_cols = 4
    padding_value = np.array(0)
    align = "LEFT_LEFT"
    input_dict = {
        "diagonal": diagonal,
        "name": name,
        "k": k,
        "num_rows": num_rows,
        "num_cols": num_cols,
        "padding_value": padding_value,
        "align": align
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    diagonal = np.array([1, 2, 3, 4])
    name = "diag5"
    k = 0
    num_rows = 4
    num_cols = 4
    padding_value = np.array(-1)
    align = "RIGHT_RIGHT"

    input_dict = {
        "diagonal": diagonal,
        "name": name,
        "k": k,
        "num_rows": num_rows,
        "num_cols": num_cols,
        "padding_value": padding_value,
        "align": align
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    diagonal = np.array([[1, 2, 3], [4, 5, 6]])
    name = "diag6"
    k = 1
    num_rows = 4
    num_cols = 4
    padding_value = np.array(0)
    align = "RIGHT_LEFT"
    input_dict = {
        "diagonal": diagonal,
        "name": name,
        "k": k,
        "num_rows": num_rows,
        "num_cols": num_cols,
        "padding_value": padding_value,
        "align": align
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    # Input 7
    diagonal = np.array([1, 2])
    name = "diag8"
    k = -1
    num_rows = 3
    num_cols = 4
    padding_value = np.array(0)
    align = "RIGHT_RIGHT"
    input_dict = {
        "diagonal": diagonal,
        "name": name,
        "k": k,
        "num_rows": num_rows,
        "num_cols": num_cols,
        "padding_value": padding_value,
        "align": align
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    diagonal = np.array([1, 2, 3, 4])
    name = "diag9"
    k = 2
    num_rows = 6
    num_cols = 6
    padding_value = np.array(-1)
    align = "LEFT_LEFT"
    input_dict = {
        "diagonal": diagonal,
        "name": name,
        "k": k,
        "num_rows": num_rows,
        "num_cols": num_cols,
        "padding_value": padding_value,
        "align": align
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.diag"] = tf_linalg_diag_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.diag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.diag'.")

check_valid('tf.linalg.diag', generated_inputs['tf.linalg.diag'], lib="tf", suffix=0)
