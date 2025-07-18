
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_tf_data_experimental_get_single_element_inputs():
    """
    Generates a list of valid inputs for the tf.data.experimental.get_single_element function.
    To satisfy the testing harness which expects .shape, .dtype, and .size attributes,
    we create a valid tf.data.Dataset and then monkey-patch these attributes onto it,
    using the values from the dataset's element_spec.
    """
    list_of_inputs = []

    def create_input(dataset):
        """Helper to create a dataset and patch its tensor-like attributes."""
        # Check if the dataset's element is a single tensor with a defined spec
        if hasattr(dataset.element_spec, 'shape') and hasattr(dataset.element_spec, 'dtype'):
            element_shape = dataset.element_spec.shape
            
            # Monkey-patch attributes for the testing framework
            dataset.shape = element_shape
            dataset.dtype = dataset.element_spec.dtype
            
            # Use num_elements() to get the size. It returns None for partially known shapes.
            num_elems = element_shape.num_elements()
            # The framework expects an integer, so provide 0 for unknown sizes.
            dataset.size = num_elems if num_elems is not None else 0
            
            return {'dataset': dataset}
        return None # Skip datasets with complex structures

    # Input 1: Dataset created by batching multiple elements into one.
    ds1 = tf.data.Dataset.from_tensor_slices(np.arange(5, dtype=np.int32)).batch(5)
    list_of_inputs.append(create_input(ds1))

    # Input 2: Dataset with a single scalar tensor.
    ds2 = tf.data.Dataset.from_tensors(np.int32(42))
    list_of_inputs.append(create_input(ds2))

    # Input 3: Dataset with a single 1D float tensor with negative values.
    ds3 = tf.data.Dataset.from_tensors(np.array([1.1, 2.2, -3.3], dtype=np.float32))
    list_of_inputs.append(create_input(ds3))

    # Input 4: Dataset with a single 2D integer tensor.
    ds4 = tf.data.Dataset.from_tensors(np.array([[-1, -2], [3, 4]], dtype=np.int16))
    list_of_inputs.append(create_input(ds4))

    # Input 5: Dataset with a single 3D float64 tensor.
    ds5 = tf.data.Dataset.from_tensors(np.ones((2, 3, 2), dtype=np.float64))
    list_of_inputs.append(create_input(ds5))

    # Input 6: Dataset with a single boolean tensor.
    ds6 = tf.data.Dataset.from_tensors(np.array([[True, False], [False, True]]))
    list_of_inputs.append(create_input(ds6))

    # Input 7: Dataset with a single string tensor.
    ds7 = tf.data.Dataset.from_tensors(np.array([b"tensorflow", b"rules"]))
    list_of_inputs.append(create_input(ds7))

    # Input 8: Dataset with a single, empty tensor.
    ds8 = tf.data.Dataset.from_tensors(np.empty((5, 0), dtype=np.int64))
    list_of_inputs.append(create_input(ds8))

    # Input 9: Dataset from batching, with a different data type (uint8).
    tensor_3d_uint = np.random.randint(0, 255, size=(4, 2, 2), dtype=np.uint8)
    ds9 = tf.data.Dataset.from_tensor_slices(tensor_3d_uint).batch(4)
    list_of_inputs.append(create_input(ds9))

    # Input 10: Dataset with a single element that is a high-rank tensor.
    high_rank_element = np.random.rand(1, 2, 1, 3, 1).astype(np.float16)
    ds10 = tf.data.Dataset.from_tensors(high_rank_element)
    list_of_inputs.append(create_input(ds10))
    
    # Input 11: Dataset with a single element of complex numbers.
    ds11 = tf.data.Dataset.from_tensors(np.array([1+2j, 3-4j], dtype=np.complex64))
    list_of_inputs.append(create_input(ds11))

    # Input 12: Dataset with a single element of shape (1,).
    ds12 = tf.data.Dataset.from_tensors(np.array([100], dtype=np.int64))
    list_of_inputs.append(create_input(ds12))

    # Filter out any None entries that might have been created
    list_of_inputs = [inp for inp in list_of_inputs if inp is not None]

    return list_of_inputs

generated_inputs["tf.data.experimental.get_single_element"] = get_tf_data_experimental_get_single_element_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.get_single_element' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.get_single_element'.")

check_valid('tf.data.experimental.get_single_element', generated_inputs['tf.data.experimental.get_single_element'], lib="tf", suffix=0)
