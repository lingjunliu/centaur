
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_autodiff_ForwardAccumulator_inputs():
    list_of_inputs = []

    # Input 1: Simple float primal and tangent
    primals = tf.constant(1.0)
    tangents = tf.constant(1.0)
    input_dict = {"primals": primals, "tangents": tangents}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Simple float primal and tangent
    primals = tf.constant(1.0)
    tangents = tf.constant(1.0)
    input_dict = {"primals": primals, "tangents": tangents}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Primal and Tangent as rank 1 tensor
    primals = tf.constant([1.0, 2.0, 3.0])
    tangents = tf.constant([0.5, 1.0, 1.5])
    input_dict = {"primals": primals, "tangents": tangents}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Primal and Tangent as rank 2 tensor
    primals = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    tangents = tf.constant([[0.5, 1.0], [1.5, 2.0]])
    input_dict = {"primals": primals, "tangents": tangents}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Primal and Tangent as rank 3 tensor
    primals = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    tangents = tf.constant([[[0.5, 1.0], [1.5, 2.0]], [[2.5, 3.0], [3.5, 4.0]]])
    input_dict = {"primals": primals, "tangents": tangents}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Primal as variable
    primals = tf.Variable(1.0)
    tangents = tf.constant(1.0)
    input_dict = {"primals": primals, "tangents": tangents}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Primal and tangent with negative values
    primals = tf.constant([-1.0, 2.0, -3.0])
    tangents = tf.constant([0.5, -1.0, 1.5])
    input_dict = {"primals": primals, "tangents": tangents}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Primal and tangent with zeros
    primals = tf.constant([0.0, 2.0, 0.0])
    tangents = tf.constant([0.0, -1.0, 0.0])
    input_dict = {"primals": primals, "tangents": tangents}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Primal and tangent with different shapes but broadcastable
    primals = tf.constant([[1.0], [2.0]])
    tangents = tf.constant([0.5, 1.0])
    input_dict = {"primals": primals, "tangents": tangents}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger primal and tangent
    primals = tf.constant(np.random.rand(10, 10).astype(np.float32))
    tangents = tf.constant(np.random.rand(10, 10).astype(np.float32))
    input_dict = {"primals": primals, "tangents": tangents}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
inputs = tf_autodiff_ForwardAccumulator_inputs()
for input_dict in inputs:
    if isinstance(input_dict["primals"], tf.Variable):
        input_dict["primals"] = input_dict["primals"].numpy()
    if isinstance(input_dict["tangents"], tf.Variable):
        input_dict["tangents"] = input_dict["tangents"].numpy()

    primals_np = input_dict["primals"]
    tangents_np = input_dict["tangents"]
    
    input_dict["primals"] = tf.convert_to_tensor(primals_np, dtype=tf.float32)
    input_dict["tangents"] = tf.convert_to_tensor(tangents_np, dtype=tf.float32)

generated_inputs["tf.autodiff.ForwardAccumulator"] = inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.autodiff.ForwardAccumulator' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.autodiff.ForwardAccumulator'.")

check_valid('tf.autodiff.ForwardAccumulator', generated_inputs['tf.autodiff.ForwardAccumulator'], lib="tf", suffix=0)
