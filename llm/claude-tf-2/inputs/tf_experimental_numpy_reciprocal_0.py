
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np

def tf_experimental_numpy_reciprocal_inputs():
    list_of_inputs = []
    
    list_of_inputs.append({"x": np.array([1, 2, 4, 8])})
    list_of_inputs.append({"x": np.array([[1.0, 2.0, 4.0], [0.5, 0.25, 0.125]])})
    list_of_inputs.append({"x": np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])})
    list_of_inputs.append({"x": np.array(5.0)})
    list_of_inputs.append({"x": np.array([-1.0, -2.0, -4.0, -8.0])})
    list_of_inputs.append({"x": np.array([0.001, 0.01, 0.1, 1.0])})
    list_of_inputs.append({"x": np.array([[-1.0, 2.0], [-3.0, 4.0]])})
    list_of_inputs.append({"x": np.array([10.0, 20.0, 30.0, 40.0, 50.0])})
    list_of_inputs.append({"x": np.array([1, 5, 10, 20], dtype=np.int32)})
    list_of_inputs.append({"x": np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)})
    list_of_inputs.append({"x": np.array([[[[1.0, 2.0], [3.0, 4.0]]]])})
    list_of_inputs.append({"x": np.array([0.25, 0.5, 0.75, 1.25])})
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.reciprocal"] = tf_experimental_numpy_reciprocal_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.reciprocal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.reciprocal'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.reciprocal', generated_inputs['tf.experimental.numpy.reciprocal'], lib="tf", suffix=0)
