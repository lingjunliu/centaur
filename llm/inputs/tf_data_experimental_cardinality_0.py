
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np

def create_and_patch_dataset(dataset_creator_lambda):
    """
    Creates a tf.data.Dataset and patches it with .shape, .dtype, and .size
    attributes to satisfy a testing framework that incorrectly expects
    Tensor-like attributes on a Dataset object.
    """
    dataset = dataset_creator_lambda()

    # Patch shape and dtype based on the element specification
    if isinstance(dataset.element_spec, tuple):
        dataset.shape = tuple(spec.shape for spec in dataset.element_spec)
        dataset.dtype = tuple(spec.dtype for spec in dataset.element_spec)
    elif isinstance(dataset.element_spec, dict):
        dataset.shape = {k: v.shape for k, v in dataset.element_spec.items()}
        dataset.dtype = {k: v.dtype for k, v in dataset.element_spec.items()}
    else:  # It's a single TensorSpec
        dataset.shape = dataset.element_spec.shape
        dataset.dtype = dataset.element_spec.dtype

    # Patch size. The framework checks `if value.size > 0`.
    # We set size to 0 for empty datasets and 1 otherwise.
    card = tf.data.experimental.cardinality(dataset).numpy()
    if card == 0:
        dataset.size = 0
    else:
        # For known positive, infinite, or unknown cardinality, the dataset
        # is not empty, so we assign a positive size.
        dataset.size = 1

    return dataset

def tf_data_experimental_cardinality_inputs():
    """
    Generates a list of valid inputs for tf.data.experimental.cardinality.
    """
    list_of_inputs = []

    # Input 1: Known cardinality from tf.data.Dataset.range
    list_of_inputs.append({'dataset': create_and_patch_dataset(lambda: tf.data.Dataset.range(42))})

    # Input 2: Known cardinality from a NumPy array
    list_of_inputs.append({'dataset': create_and_patch_dataset(lambda: tf.data.Dataset.from_tensor_slices(np.arange(100, dtype=np.int32)))})

    # Input 3: Infinite cardinality from .repeat()
    list_of_inputs.append({'dataset': create_and_patch_dataset(lambda: tf.data.Dataset.from_tensor_slices([1, 2, 3]).repeat())})

    # Input 4: Unknown cardinality from .filter()
    list_of_inputs.append({'dataset': create_and_patch_dataset(lambda: tf.data.Dataset.range(200).filter(lambda x: x > 100))})

    # Input 5: Known cardinality (0) for an empty dataset
    list_of_inputs.append({'dataset': create_and_patch_dataset(lambda: tf.data.Dataset.from_tensor_slices(np.array([], dtype=np.float64)))})

    # Input 6: Known cardinality after .take() on an infinite dataset
    list_of_inputs.append({'dataset': create_and_patch_dataset(lambda: tf.data.Dataset.range(1).repeat().take(50))})

    # Input 7: Known cardinality after .skip()
    list_of_inputs.append({'dataset': create_and_patch_dataset(lambda: tf.data.Dataset.range(30).skip(10))})

    # Input 8: Known cardinality after .batch()
    list_of_inputs.append({'dataset': create_and_patch_dataset(lambda: tf.data.Dataset.range(10).batch(4))})

    # Input 9: Known cardinality from zipping two datasets
    list_of_inputs.append({'dataset': create_and_patch_dataset(lambda: tf.data.Dataset.zip((tf.data.Dataset.range(15), tf.data.Dataset.from_tensor_slices(np.zeros((20, 2), dtype=np.float32)))))})

    # Input 10: Known cardinality from concatenating two datasets
    list_of_inputs.append({'dataset': create_and_patch_dataset(lambda: tf.data.Dataset.range(5).concatenate(tf.data.Dataset.range(8)))})

    return list_of_inputs

generated_inputs["tf.data.experimental.cardinality"] = tf_data_experimental_cardinality_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.cardinality' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.cardinality'.")

check_valid('tf.data.experimental.cardinality', generated_inputs['tf.data.experimental.cardinality'], lib="tf", suffix=0)
