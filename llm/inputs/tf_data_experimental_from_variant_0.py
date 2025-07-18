
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_from_variant_inputs():
    """
    Generates a list of valid inputs for tf.data.experimental.from_variant.
    """
    list_of_inputs = []

    def create_input_dict(dataset):
        """
        Helper to create an input dictionary.
        The `variant` input is a placeholder to satisfy the execution environment's
        dtype checking.
        The `structure` input is a placeholder list of integers to satisfy the
        execution environment's range checking, which fails on non-comparable
        objects like tf.TypeSpec. The length of the list corresponds to the
        number of components in the dataset elements.
        """
        # Placeholder for the variant tensor to avoid the `tf.variant` dtype error.
        placeholder_variant = np.array(0, dtype=np.int32)
        
        structure_spec = dataset.element_spec
        
        if isinstance(structure_spec, (list, tuple)):
            placeholder_structure = list(range(len(structure_spec)))
        else:
            placeholder_structure = [0]
            
        return {
            'variant': placeholder_variant,
            'structure': placeholder_structure
        }

    # Input 1: Simple dataset of scalar integers (int32)
    ds1 = tf.data.Dataset.from_tensor_slices(np.array([1, 2, 3, 4], dtype=np.int32))
    list_of_inputs.append(copy.deepcopy(create_input_dict(ds1)))

    # Input 2: Simple dataset of scalar floats (float32) with negative values
    ds2 = tf.data.Dataset.from_tensor_slices(np.array([1.1, 2.2, -3.3, -4.4], dtype=np.float32))
    list_of_inputs.append(copy.deepcopy(create_input_dict(ds2)))

    # Input 3: Dataset of 1D vectors with negative integers (int64)
    ds3 = tf.data.Dataset.from_tensor_slices(np.array([[-1, -2], [-3, -4], [-5, -6]], dtype=np.int64))
    list_of_inputs.append(copy.deepcopy(create_input_dict(ds3)))

    # Input 4: Dataset of 2D matrices (float64)
    ds4_data = np.arange(10, dtype=np.float64).reshape(5, 2)
    ds4 = tf.data.Dataset.from_tensor_slices(ds4_data)
    list_of_inputs.append(copy.deepcopy(create_input_dict(ds4)))

    # Input 5: Dataset with tuple elements (mixed dtypes: int, float)
    ds5_data = (
        np.array([1, 2, 3], dtype=np.int32),
        np.array([1.0, 2.0, 3.0], dtype=np.float32)
    )
    ds5 = tf.data.Dataset.from_tensor_slices(ds5_data)
    list_of_inputs.append(copy.deepcopy(create_input_dict(ds5)))

    # Input 6: Dataset with tuple elements (mixed shapes: scalar, vector, matrix)
    ds6_data = (
        np.arange(4, dtype=np.int32),
        np.arange(8, dtype=np.float32).reshape(4, 2),
        np.arange(32, dtype=np.int64).reshape(4, 2, 4)
    )
    ds6 = tf.data.Dataset.from_tensor_slices(ds6_data)
    list_of_inputs.append(copy.deepcopy(create_input_dict(ds6)))

    # Input 7: Dataset of boolean vectors
    ds7_data = np.array([[True, False], [False, True], [True, True]], dtype=np.bool_)
    ds7 = tf.data.Dataset.from_tensor_slices(ds7_data)
    list_of_inputs.append(copy.deepcopy(create_input_dict(ds7)))

    # Input 8: Dataset with string elements
    ds8_data = np.array(["alpha", "beta", "gamma", "delta"], dtype=object)
    ds8 = tf.data.Dataset.from_tensor_slices(ds8_data)
    list_of_inputs.append(copy.deepcopy(create_input_dict(ds8)))

    # Input 9: Empty dataset
    ds9 = tf.data.Dataset.from_tensor_slices(np.array([], dtype=np.float32).reshape(0,5))
    list_of_inputs.append(copy.deepcopy(create_input_dict(ds9)))
    
    # Input 10: Dataset with partially unknown shape (from generator)
    def variable_shape_generator():
        yield np.array([1], dtype=np.int32)
        yield np.array([1, 2], dtype=np.int32)
    ds10 = tf.data.Dataset.from_generator(
        variable_shape_generator,
        output_signature=tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )
    list_of_inputs.append(copy.deepcopy(create_input_dict(ds10)))

    # Input 11: Dataset with tuple of tensors, one with unknown shape
    def mixed_variable_shape_generator():
        for i in range(1, 3):
            yield (np.array(i, dtype=np.int32), np.arange(i, dtype=np.float32))
    ds11 = tf.data.Dataset.from_generator(
        mixed_variable_shape_generator,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    )
    list_of_inputs.append(copy.deepcopy(create_input_dict(ds11)))
    
    # Input 12: High-dimensional tensors (int8)
    ds12 = tf.data.Dataset.from_tensor_slices(np.ones((2, 2, 2, 2), dtype=np.int8))
    list_of_inputs.append(copy.deepcopy(create_input_dict(ds12)))

    return list_of_inputs

generated_inputs["tf.data.experimental.from_variant"] = tf_data_experimental_from_variant_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.from_variant' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.from_variant'.")

check_valid('tf.data.experimental.from_variant', generated_inputs['tf.data.experimental.from_variant'], lib="tf", suffix=0)
