
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_clip_by_value_inputs():
    list_of_inputs = []

    # Input 1: Basic usage with scalar min/max
    t = tf.constant(np.array([[-10., -1., 0.], [0., 2., 10.]]))
    clip_value_min = tf.constant(-1.0)
    clip_value_max = tf.constant(1.0)
    name = "clip_basic"
    input_dict = {"t": t, "clip_value_min": clip_value_min.numpy(), "clip_value_max": clip_value_max.numpy(), "name": name, "t_dtype": str(t.dtype)}
    input_dict["t"] = t.numpy()
    input_dict["clip_value_min"] = clip_value_min.numpy()
    input_dict["clip_value_max"] = clip_value_max.numpy()
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Broadcasting min/max
    t = tf.constant(np.array([[-1, 0., 10.], [-1, 0, 10.]]))
    clip_value_min = tf.constant(np.array([[2.],[1.]]))
    clip_value_max = tf.constant(100.0)
    name = "clip_broadcast"
    input_dict = {"t": t, "clip_value_min": clip_value_min.numpy(), "clip_value_max": clip_value_max.numpy(), "name": name, "t_dtype": str(t.dtype)}
    input_dict["t"] = t.numpy()
    input_dict["clip_value_min"] = clip_value_min.numpy()
    input_dict["clip_value_max"] = clip_value_max.numpy()
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative values in min/max
    t = tf.constant(np.array([[-5, 0, 5], [-10, 0, 10]]))
    clip_value_min = tf.constant(-3)
    clip_value_max = tf.constant(-1)
    name = "clip_negative"
    input_dict = {"t": t, "clip_value_min": clip_value_min.numpy(), "clip_value_max": clip_value_max.numpy(), "name": name, "t_dtype": str(t.dtype)}
    input_dict["t"] = t.numpy()
    input_dict["clip_value_min"] = clip_value_min.numpy()
    input_dict["clip_value_max"] = clip_value_max.numpy()
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Same min and max
    t = tf.constant(np.array([[-1, 0, 1], [-2, 0, 2]]))
    clip_value_min = tf.constant(0)
    clip_value_max = tf.constant(0)
    name = "clip_same"
    input_dict = {"t": t, "clip_value_min": clip_value_min.numpy(), "clip_value_max": clip_value_max.numpy(), "name": name, "t_dtype": str(t.dtype)}
    input_dict["t"] = t.numpy()
    input_dict["clip_value_min"] = clip_value_min.numpy()
    input_dict["clip_value_max"] = clip_value_max.numpy()
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D tensor
    t = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    clip_value_min = tf.constant(2)
    clip_value_max = tf.constant(7)
    name = "clip_3d"
    input_dict = {"t": t, "clip_value_min": clip_value_min.numpy(), "clip_value_max": clip_value_max.numpy(), "name": name, "t_dtype": str(t.dtype)}
    input_dict["t"] = t.numpy()
    input_dict["clip_value_min"] = clip_value_min.numpy()
    input_dict["clip_value_max"] = clip_value_max.numpy()
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Float type tensors
    t = tf.constant(np.array([[-1.5, 0.0, 1.5], [-2.5, 0.0, 2.5]]))
    clip_value_min = tf.constant(-1.0)
    clip_value_max = tf.constant(1.0)
    name = "clip_float"
    input_dict = {"t": t, "clip_value_min": clip_value_min.numpy(), "clip_value_max": clip_value_max.numpy(), "name": name, "t_dtype": str(t.dtype)}
    input_dict["t"] = t.numpy()
    input_dict["clip_value_min"] = clip_value_min.numpy()
    input_dict["clip_value_max"] = clip_value_max.numpy()
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Rank 4 Tensor
    t = tf.constant(np.random.rand(2, 3, 4, 5))
    clip_value_min = tf.constant(0.2)
    clip_value_max = tf.constant(0.8)
    name = "clip_rank4"
    input_dict = {"t": t, "clip_value_min": clip_value_min.numpy(), "clip_value_max": clip_value_max.numpy(), "name": name, "t_dtype": str(t.dtype)}
    input_dict["t"] = t.numpy()
    input_dict["clip_value_min"] = clip_value_min.numpy()
    input_dict["clip_value_max"] = clip_value_max.numpy()
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Broadcasting with different shape
    t = tf.constant(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64))
    clip_value_min = tf.constant(np.array([[1],[2]], dtype=np.float64))
    clip_value_max = tf.constant(np.array([[4],[5]], dtype=np.float64))
    name = "clip_broadcast_2"
    input_dict = {"t": t, "clip_value_min": clip_value_min.numpy(), "clip_value_max": clip_value_max.numpy(), "name": name, "t_dtype": str(t.dtype)}
    input_dict["t"] = t.numpy()
    input_dict["clip_value_min"] = clip_value_min.numpy()
    input_dict["clip_value_max"] = clip_value_max.numpy()
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: All values outside range
    t = tf.constant(np.array([[-5, -4, -3], [3, 4, 5]]))
    clip_value_min = tf.constant(-2)
    clip_value_max = tf.constant(2)
    name = "clip_outside"
    input_dict = {"t": t, "clip_value_min": clip_value_min.numpy(), "clip_value_max": clip_value_max.numpy(), "name": name, "t_dtype": str(t.dtype)}
    input_dict["t"] = t.numpy()
    input_dict["clip_value_min"] = clip_value_min.numpy()
    input_dict["clip_value_max"] = clip_value_max.numpy()
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10: int32 tensor
    t = tf.constant(np.array([[-5, -4, -3], [3, 4, 5]], dtype=np.int32))
    clip_value_min = tf.constant(-2, dtype=np.int32)
    clip_value_max = tf.constant(2, dtype=np.int32)
    name = "clip_int32"
    input_dict = {"t": t, "clip_value_min": clip_value_min.numpy(), "clip_value_max": clip_value_max.numpy(), "name": name, "t_dtype": str(t.dtype)}
    input_dict["t"] = t.numpy()
    input_dict["clip_value_min"] = clip_value_min.numpy()
    input_dict["clip_value_max"] = clip_value_max.numpy()
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.clip_by_value"] = tf_clip_by_value_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.clip_by_value' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.clip_by_value'.")

check_valid('tf.clip_by_value', generated_inputs['tf.clip_by_value'], lib="tf", suffix=0)
