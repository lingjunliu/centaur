
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

# This helper class is designed to satisfy multiple conflicting requirements from
# the testing harness and the TensorFlow API.
# 1. API Requirement: The 'shape' argument must be an `isinstance` of
#    `tf.TensorShape`. Solved by inheriting from `tf.TensorShape`.
# 2. Harness Requirements: The input object must have `.shape`, `.dtype`, and
#    `.size` attributes, similar to a tf.Tensor.
# 3. Harness Behavior: The harness uses `copy.deepcopy`, which by default strips
#    custom attributes from `tf.TensorShape` objects.
#
# This class implements the necessary attributes and a custom `__deepcopy__`
# method to ensure they are preserved during the copy operation.
class PatchedTensorShape(tf.TensorShape):
    def __init__(self, dims):
        super().__init__(dims)
        # Add attributes required by the testing harness.
        # Tensor shapes are conceptually 1D integer lists, so their "shape" is
        # (rank,), their "size" is rank, and their "dtype" is int.
        if self.rank is None:
            self.shape = tuple()
            self.size = 0
        else:
            self.shape = (self.rank,)
            self.size = self.rank
        self.dtype = tf.int32

    def __deepcopy__(self, memo):
        # The default deepcopy for tf.TensorShape does not preserve custom
        # attributes. We override it to create a new instance of our own class,
        # which will re-run __init__ and correctly set the custom attributes.
        new_obj = self.__class__(self.as_list() if self.rank is not None else None)
        memo[id(self)] = new_obj
        return new_obj

def get_tf_compat_dimension_at_index_inputs():
    list_of_inputs = []

    # Input 1: Simple 3D shape, positive index
    input_dict = {
        'shape': PatchedTensorShape([2, 3, 4]),
        'index': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D shape, index 0
    input_dict = {
        'shape': PatchedTensorShape([10, 20]),
        'index': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 4D shape, last positive index
    input_dict = {
        'shape': PatchedTensorShape([5, 6, 7, 8]),
        'index': 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D shape, negative index -1
    input_dict = {
        'shape': PatchedTensorShape([8, 7, 6]),
        'index': -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 5D shape, other negative index
    input_dict = {
        'shape': PatchedTensorShape([1, 2, 3, 4, 5]),
        'index': -3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Shape with an unknown dimension (None), accessing the unknown dim
    input_dict = {
        'shape': PatchedTensorShape([10, None, 5]),
        'index': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Shape with an unknown dimension (None), accessing a known dim
    input_dict = {
        'shape': PatchedTensorShape([None, 20, 30]),
        'index': 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D shape (vector), index 0
    input_dict = {
        'shape': PatchedTensorShape([100]),
        'index': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Shape containing a zero dimension
    input_dict = {
        'shape': PatchedTensorShape([64, 0, 128]),
        'index': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: High rank shape
    input_dict = {
        'shape': PatchedTensorShape([2, 3, 2, 3, 2, 3]),
        'index': 5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.compat.dimension_at_index"] = get_tf_compat_dimension_at_index_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.compat.dimension_at_index' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.compat.dimension_at_index'.")

check_valid('tf.compat.dimension_at_index', generated_inputs['tf.compat.dimension_at_index'], lib="tf", suffix=0)
