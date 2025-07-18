
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_from_variant_inputs():
    list_of_inputs = []

    def create_variant_numpy_scalar(dataset):
        variant_tensor = tf.data.experimental.to_variant(dataset)
        # Wrap the variant tensor in a 0-D numpy object array to conform to the
        # "numpy format" requirement, as tf.variant tensors cannot be directly
        # converted to numpy arrays.
        scalar_np_object_array = np.empty((), dtype=object)
        scalar_np_object_array[()] = variant_tensor
        return scalar_np_object_array

    # Input 1: Dataset of empty tuples from_tensor_slices
    dataset1 = tf.data.Dataset.from_tensor_slices([()] * 5)
    variant1 = create_variant_numpy_scalar(dataset1)
    # The element_spec for a dataset of empty tuples is `()`. list(()) is [].
    # This satisfies the testing environment's constraints.
    structure1 = list(dataset1.element_spec)
    list_of_inputs.append(copy.deepcopy({
        'variant': variant1,
        'structure': structure1
    }))

    # Input 2: Dataset with a single empty tuple element from_tensors
    # On some TF versions, from_tensors with a list can create a single element
    # of a variant tensor, whose spec is a non-iterable TensorSpec.
    # Using from_tensors with an empty tuple creates an element_spec of `()`, which is iterable.
    dataset2 = tf.data.Dataset.from_tensors(())
    variant2 = create_variant_numpy_scalar(dataset2)
    structure2 = list(dataset2.element_spec)
    list_of_inputs.append(copy.deepcopy({
        'variant': variant2,
        'structure': structure2
    }))

    # Input 3: Dataset using range and map to empty tuples
    dataset3 = tf.data.Dataset.range(8).map(lambda x: ())
    variant3 = create_variant_numpy_scalar(dataset3)
    structure3 = list(dataset3.element_spec)
    list_of_inputs.append(copy.deepcopy({
        'variant': variant3,
        'structure': structure3
    }))

    # Input 4: Empty dataset of empty tuples
    dataset4 = tf.data.Dataset.from_tensor_slices([()] * 0)
    variant4 = create_variant_numpy_scalar(dataset4)
    structure4 = list(dataset4.element_spec)
    list_of_inputs.append(copy.deepcopy({
        'variant': variant4,
        'structure': structure4
    }))
    
    # Input 5: Zipping two datasets and mapping to empty tuple
    ds1 = tf.data.Dataset.range(5)
    ds2 = tf.data.Dataset.from_tensor_slices(np.arange(5))
    dataset5 = tf.data.Dataset.zip((ds1, ds2)).map(lambda x, y: ())
    variant5 = create_variant_numpy_scalar(dataset5)
    structure5 = list(dataset5.element_spec)
    list_of_inputs.append(copy.deepcopy({
        'variant': variant5,
        'structure': structure5
    }))

    # Input 6: Batching and then mapping to empty tuple
    dataset6 = tf.data.Dataset.range(10).batch(2).map(lambda x: ())
    variant6 = create_variant_numpy_scalar(dataset6)
    structure6 = list(dataset6.element_spec)
    list_of_inputs.append(copy.deepcopy({
        'variant': variant6,
        'structure': structure6
    }))
    
    # Input 7: Using from_generator to yield empty tuples
    def empty_gen():
        for _ in range(7):
            yield ()
    dataset7 = tf.data.Dataset.from_generator(empty_gen, output_signature=())
    variant7 = create_variant_numpy_scalar(dataset7)
    structure7 = list(dataset7.element_spec)
    list_of_inputs.append(copy.deepcopy({
        'variant': variant7,
        'structure': structure7
    }))

    # Input 8: Another from_tensor_slices with different length
    dataset8 = tf.data.Dataset.from_tensor_slices([()] * 3)
    variant8 = create_variant_numpy_scalar(dataset8)
    structure8 = list(dataset8.element_spec)
    list_of_inputs.append(copy.deepcopy({
        'variant': variant8,
        'structure': structure8
    }))
    
    # Input 9: Using tf.data.Dataset.repeat
    dataset9 = tf.data.Dataset.from_tensor_slices([()]).repeat(4)
    variant9 = create_variant_numpy_scalar(dataset9)
    structure9 = list(dataset9.element_spec)
    list_of_inputs.append(copy.deepcopy({
        'variant': variant9,
        'structure': structure9
    }))
    
    # Input 10: Using tf.data.Dataset.take
    dataset10 = tf.data.Dataset.range(100).map(lambda x: ()).take(12)
    variant10 = create_variant_numpy_scalar(dataset10)
    structure10 = list(dataset10.element_spec)
    list_of_inputs.append(copy.deepcopy({
        'variant': variant10,
        'structure': structure10
    }))

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
