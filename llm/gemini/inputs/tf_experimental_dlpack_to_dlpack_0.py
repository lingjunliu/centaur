
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_dlpack_to_dlpack_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 tensor
    tf_tensor = tf.constant([1.0, 2.0, 3.0, 4.0], dtype=tf.float32)
    list_of_inputs.append(copy.deepcopy({"tf_tensor": tf_tensor}))

    # Input 2: 2D int32 tensor with negative values
    tf_tensor = tf.constant([[-1, 2], [3, -4]], dtype=tf.int32)
    list_of_inputs.append(copy.deepcopy({"tf_tensor": tf_tensor}))

    # Input 3: 3D float64 tensor
    tf_tensor = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=tf.float64)
    list_of_inputs.append(copy.deepcopy({"tf_tensor": tf_tensor}))

    # Input 4: 0D (scalar) float32 tensor
    tf_tensor = tf.constant(42.0, dtype=tf.float32)
    list_of_inputs.append(copy.deepcopy({"tf_tensor": tf_tensor}))

    # Input 5: 4D int64 tensor
    tf_tensor = tf.constant([[[[1, 2]], [[3, 4]]]], dtype=tf.int64)
    list_of_inputs.append(copy.deepcopy({"tf_tensor": tf_tensor}))

    # Input 6: 1D float16 tensor
    tf_tensor = tf.constant([0.1, -0.2, 0.3], dtype=tf.float16)
    list_of_inputs.append(copy.deepcopy({"tf_tensor": tf_tensor}))

    # Input 7: 2D uint8 tensor
    tf_tensor = tf.constant([[0, 255], [128, 64]], dtype=tf.uint8)
    list_of_inputs.append(copy.deepcopy({"tf_tensor": tf_tensor}))

    # Input 8: 3D int8 tensor
    tf_tensor = tf.constant([[[1, -1]], [[2, -2]]], dtype=tf.int8)
    list_of_inputs.append(copy.deepcopy({"tf_tensor": tf_tensor}))

    # Input 9: Large 1D float32 tensor
    tf_tensor = tf.random.uniform(shape=[100], minval=-1.0, maxval=1.0, dtype=tf.float32)
    list_of_inputs.append(copy.deepcopy({"tf_tensor": tf_tensor}))

    # Input 10: 5D float32 tensor
    tf_tensor = tf.zeros(shape=(2, 2, 2, 2, 2), dtype=tf.float32)
    list_of_inputs.append(copy.deepcopy({"tf_tensor": tf_tensor}))

    return list_of_inputs

generated_inputs["tf.experimental.dlpack.to_dlpack"] = tf_experimental_dlpack_to_dlpack_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.dlpack.to_dlpack' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.dlpack.to_dlpack'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.dlpack.to_dlpack', generated_inputs['tf.experimental.dlpack.to_dlpack'], lib="tf", suffix=0)
