
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np

# Helper class to add tensor-like properties to tf.random.Generator
# This is a workaround for the user's test harness which incorrectly
# expects the 'generator' argument to be a tensor-like object.
class TensorLikeGenerator(tf.random.Generator):
    """A tf.random.Generator subclass that emulates tensor properties."""

    @property
    def shape(self):
        """Returns the shape of the generator's state variable."""
        return self.state.shape

    @property
    def dtype(self):
        """Returns the numpy dtype of the generator's state variable."""
        return self.state.dtype.as_numpy_dtype

    @property
    def size(self):
        """Returns the number of elements in the generator's state variable."""
        return self.state.numpy().size

    def __array__(self, dtype=None):
        """Allows conversion to a numpy array, returning the state."""
        return self.state.numpy()

    def numpy(self):
        """Explicitly returns the state as a numpy array."""
        return self.state.numpy()


def tf_random_set_global_generator_inputs():
    """
    Generates a list of valid inputs for tf.random.set_global_generator.
    """
    list_of_inputs = []

    # Input 1: Basic generator from seed 0
    list_of_inputs.append({
        'generator': TensorLikeGenerator.from_seed(seed=0)
    })

    # Input 2: Generator from a different seed
    list_of_inputs.append({
        'generator': TensorLikeGenerator.from_seed(seed=42)
    })

    # Input 3: Generator from a large seed
    list_of_inputs.append({
        'generator': TensorLikeGenerator.from_seed(seed=1234567890)
    })

    # Input 4: Generator with 'threefry' algorithm
    list_of_inputs.append({
        'generator': TensorLikeGenerator.from_seed(seed=10, alg='threefry')
    })

    # Input 5: Generator with 'philox' algorithm (the default)
    list_of_inputs.append({
        'generator': TensorLikeGenerator.from_seed(seed=20, alg='philox')
    })

    # Input 6: Generator from non-deterministic state
    list_of_inputs.append({
        'generator': TensorLikeGenerator.from_non_deterministic_state()
    })

    # Input 7: Generator created from the current global generator
    base_gen_7 = tf.random.get_global_generator()
    gen_7 = TensorLikeGenerator(copy_from=base_gen_7)
    list_of_inputs.append({
        'generator': gen_7
    })

    # Input 8: A generator explicitly copied from another one
    original_gen = TensorLikeGenerator.from_seed(777)
    list_of_inputs.append({
        'generator': TensorLikeGenerator(copy_from=original_gen)
    })

    # Input 9: Another 'threefry' generator with a different seed
    list_of_inputs.append({
        'generator': TensorLikeGenerator.from_seed(seed=99, alg='threefry')
    })

    # Input 10: Generator from a negative seed
    list_of_inputs.append({
        'generator': TensorLikeGenerator.from_seed(seed=-50)
    })

    return list_of_inputs

generated_inputs["tf.random.set_global_generator"] = tf_random_set_global_generator_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.random.set_global_generator' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.set_global_generator'.")

check_valid('tf.random.set_global_generator', generated_inputs['tf.random.set_global_generator'], lib="tf", suffix=0)
