
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_reducer_inputs():
    list_of_inputs = []

    # Input 1
    def init_func_1(_):
        return np.float32(0.0)

    def reduce_func_1(state, value):
        return np.float32(state + value)

    def finalize_func_1(state):
        return np.float32(state)

    input_dict = {
        "init_func": [init_func_1],
        "reduce_func": [reduce_func_1],
        "finalize_func": [finalize_func_1]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    def init_func_2(_):
        return np.array([0, 0], dtype=np.int32)

    def reduce_func_2(state, value):
        return state + value

    def finalize_func_2(state):
        return np.float32(state[0] / state[1] if state[1] != 0 else 0)

    input_dict = {
        "init_func": [init_func_2],
        "reduce_func": [reduce_func_2],
        "finalize_func": [finalize_func_2]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    def init_func_3(_):
        return ""

    def reduce_func_3(state, value):
        return state + str(value)

    def finalize_func_3(state):
        return state

    input_dict = {
        "init_func": [init_func_3],
        "reduce_func": [reduce_func_3],
        "finalize_func": [finalize_func_3]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    def init_func_4(_):
        return np.array([[0, 0], [0, 0]], dtype=np.int32)

    def reduce_func_4(state, value):
        return state + value

    def finalize_func_4(state):
        return np.sum(state)

    input_dict = {
        "init_func": [init_func_4],
        "reduce_func": [reduce_func_4],
        "finalize_func": [finalize_func_4]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    def init_func_5(_):
        return np.int32(0)

    def reduce_func_5(state, value):
        return np.int32(state + value)

    def finalize_func_5(state):
        return np.int32(state * 2)

    input_dict = {
        "init_func": [init_func_5],
        "reduce_func": [reduce_func_5],
        "finalize_func": [finalize_func_5]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    def init_func_6(_):
        return np.float64(1.0)

    def reduce_func_6(state, value):
        return np.float64(state * value)

    def finalize_func_6(state):
        return np.float64(np.log(state))

    input_dict = {
        "init_func": [init_func_6],
        "reduce_func": [reduce_func_6],
        "finalize_func": [finalize_func_6]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    def init_func_7(_):
        return []

    def reduce_func_7(state, value):
        state.append(value)
        return state

    def finalize_func_7(state):
        return len(state)

    input_dict = {
        "init_func": [init_func_7],
        "reduce_func": [reduce_func_7],
        "finalize_func": [finalize_func_7]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    def init_func_8(_):
        return [0, 0]

    def reduce_func_8(state, value):
        state[0] += value
        state[1] += 1
        return state

    def finalize_func_8(state):
        return state[0] / state[1] if state[1] > 0 else 0

    input_dict = {
        "init_func": [init_func_8],
        "reduce_func": [reduce_func_8],
        "finalize_func": [finalize_func_8]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    def init_func_9(_):
        return {"sum": 0, "count": 0}

    def reduce_func_9(state, value):
        state["sum"] += value
        state["count"] += 1
        return state

    def finalize_func_9(state):
        return state["sum"] / state["count"] if state["count"] > 0 else 0

    input_dict = {
        "init_func": [init_func_9],
        "reduce_func": [reduce_func_9],
        "finalize_func": [finalize_func_9]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    def init_func_10(_):
        return np.array([0.0, 0.0], dtype=np.float32)

    def reduce_func_10(state, value):
        return np.array([state[0] + value['features'], state[1] + 1.0], dtype=np.float32)

    def finalize_func_10(state):
        return state[0] / state[1]

    input_dict = {
        "init_func": [init_func_10],
        "reduce_func": [reduce_func_10],
        "finalize_func": [finalize_func_10]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.Reducer"] = tf_data_experimental_reducer_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.Reducer' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.Reducer'.")

check_valid('tf.data.experimental.Reducer', generated_inputs['tf.data.experimental.Reducer'], lib="tf", suffix=0)
