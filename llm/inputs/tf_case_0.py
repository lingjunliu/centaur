
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_case_inputs():
    list_of_inputs = []

    # Input 1: Basic case with default
    pred_fn_pairs = [(tf.constant(False), lambda: [np.array(17, dtype=np.int32)])]
    default = lambda: [np.array(23, dtype=np.int32)]
    exclusive = False
    strict = False
    name = "case_example_1"

    input_dict = {
        "pred_fn_pairs": pred_fn_pairs,
        "default": default,
        "exclusive": exclusive,
        "strict": strict,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Predicate is true
    pred_fn_pairs = [(tf.constant(True), lambda: [np.array(17, dtype=np.int32)])]
    default = lambda: [np.array(23, dtype=np.int32)]
    exclusive = False
    strict = False
    name = "case_example_2"

    input_dict = {
        "pred_fn_pairs": pred_fn_pairs,
        "default": default,
        "exclusive": exclusive,
        "strict": strict,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multiple predicates, only one true, exclusive=False
    pred_fn_pairs = [(tf.constant(False), lambda: [np.array(17, dtype=np.int32)]), (tf.constant(True), lambda: [np.array(23, dtype=np.int32)])]
    default = lambda: [np.array(-1, dtype=np.int32)]
    exclusive = False
    strict = False
    name = "case_example_3"

    input_dict = {
        "pred_fn_pairs": pred_fn_pairs,
        "default": default,
        "exclusive": exclusive,
        "strict": strict,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multiple predicates, none true, exclusive=False
    pred_fn_pairs = [(tf.constant(False), lambda: [np.array(17, dtype=np.int32)]), (tf.constant(False), lambda: [np.array(23, dtype=np.int32)])]
    default = lambda: [np.array(-1, dtype=np.int32)]
    exclusive = False
    strict = False
    name = "case_example_4"

    input_dict = {
        "pred_fn_pairs": pred_fn_pairs,
        "default": default,
        "exclusive": exclusive,
        "strict": strict,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multiple predicates, exclusive=True, none true
    pred_fn_pairs = [(tf.constant(False), lambda: [np.array(17, dtype=np.int32)]), (tf.constant(False), lambda: [np.array(23, dtype=np.int32)])]
    default = lambda: [np.array(-1, dtype=np.int32)]
    exclusive = True
    strict = False
    name = "case_example_5"

    input_dict = {
        "pred_fn_pairs": pred_fn_pairs,
        "default": default,
        "exclusive": exclusive,
        "strict": strict,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Nested structures
    pred_fn_pairs = [(tf.constant(True), lambda: [[np.array(17, dtype=np.int32), np.array(18, dtype=np.int32)], (np.array(19, dtype=np.int32), np.array(20, dtype=np.int32))])]
    default = lambda: [[np.array(23, dtype=np.int32), np.array(24, dtype=np.int32)], (np.array(25, dtype=np.int32), np.array(26, dtype=np.int32))]
    exclusive = False
    strict = False
    name = "case_example_6"

    input_dict = {
        "pred_fn_pairs": pred_fn_pairs,
        "default": default,
        "exclusive": exclusive,
        "strict": strict,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Strict mode enabled (singletons)
    pred_fn_pairs = [(tf.constant(True), lambda: [np.array(17, dtype=np.int32)])]
    default = lambda: [np.array(23, dtype=np.int32)]
    exclusive = False
    strict = True
    name = "case_example_7"

    input_dict = {
        "pred_fn_pairs": pred_fn_pairs,
        "default": default,
        "exclusive": exclusive,
        "strict": strict,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Multiple outputs
    pred_fn_pairs = [(tf.constant(True), lambda: [np.array(17, dtype=np.int32), np.array(18, dtype=np.int32)])]
    default = lambda: [np.array(23, dtype=np.int32), np.array(24, dtype=np.int32)]
    exclusive = False
    strict = False
    name = "case_example_8"

    input_dict = {
        "pred_fn_pairs": pred_fn_pairs,
        "default": default,
        "exclusive": exclusive,
        "strict": strict,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Different tensor shapes
    pred_fn_pairs = [(tf.constant(True), lambda: [np.array([[1, 2], [3, 4]], dtype=np.int32)])]
    default = lambda: [np.array([[5, 6], [7, 8]], dtype=np.int32)]
    exclusive = False
    strict = False
    name = "case_example_9"

    input_dict = {
        "pred_fn_pairs": pred_fn_pairs,
        "default": default,
        "exclusive": exclusive,
        "strict": strict,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 10: Empty pred_fn_pairs, requires default
    pred_fn_pairs = []
    default = lambda: [np.array(42, dtype=np.int32)]
    exclusive = False
    strict = False
    name = "case_example_10"

    input_dict = {
        "pred_fn_pairs": pred_fn_pairs,
        "default": default,
        "exclusive": exclusive,
        "strict": strict,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    for input_dict in list_of_inputs:
        input_dict["pred_fn_pairs"] = [(tf.constant(x[0].numpy().item()), lambda y=x[1]: [np.array(z, dtype=np.int32) if not isinstance(z, list) and not isinstance(z, tuple) else [np.array(k, dtype=np.int32) for k in z] for z in y()]) for x in input_dict["pred_fn_pairs"]]
        input_dict["default"] = lambda y=input_dict["default"]: [np.array(z, dtype=np.int32) if not isinstance(z, list) and not isinstance(z, tuple) else [np.array(k, dtype=np.int32) for k in z] for z in y()]
        input_dict["pred_fn_pairs"] = [(x[0], x[1]) for x in input_dict["pred_fn_pairs"]] # Ensure lambdas are properly re-associated.
        input_dict["default"] = input_dict["default"]

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.case"] = tf_case_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.case' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.case'.")

check_valid('tf.case', generated_inputs['tf.case'], lib="tf", suffix=0)
