
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

# Define a custom SparseTensor class to be compatible with the analysis script
# that expects numpy-like objects with .size and compatibility with np.min/max.
class AnalysableSparseTensor(tf.sparse.SparseTensor):
    """
    A subclass of tf.sparse.SparseTensor that adds compatibility for an
    analysis script that expects .size and works with np.min/np.max.
    """
    def __array__(self, dtype=None):
        """
        This hook allows numpy functions like np.min/np.max to work on this object.
        They will operate on the .values tensor.
        """
        return tf.convert_to_tensor(self.values).numpy()

    @property
    def size(self):
        """
        The analysis script expects a .size attribute.
        We define it as the number of non-zero elements.
        """
        return tf.size(self.values)

def tf_edit_distance_inputs():
    """
    Generates a list of valid inputs for the tf.edit_distance function.
    """
    list_of_inputs = []

    def create_sparse_tensor(sequences, dense_shape, dtype=np.int64):
        indices = []
        values = []
        rank = len(dense_shape)

        if rank == 2:
            for i, seq in enumerate(sequences):
                for j, val in enumerate(seq):
                    indices.append([i, j])
                    values.append(val)
            empty_indices_shape = (0, 2)
        elif rank == 3:
            for i in range(dense_shape[0]):
                for j in range(dense_shape[1]):
                    if i < len(sequences) and j < len(sequences[i]):
                        seq = sequences[i][j]
                        for k, val in enumerate(seq):
                            indices.append([i, j, k])
                            values.append(val)
            empty_indices_shape = (0, 3)
        else:
            raise ValueError(f"Unsupported rank for this helper: {rank}")

        if not values:
            return AnalysableSparseTensor(
                indices=np.empty(empty_indices_shape, dtype=np.int64),
                values=np.array([], dtype=dtype),
                dense_shape=dense_shape)
        
        return AnalysableSparseTensor(
            indices=np.array(indices, dtype=np.int64),
            values=np.array(values, dtype=dtype),
            dense_shape=dense_shape)

    s2i = lambda s: [ord(c) for c in s]

    # Input 1: Basic Rank-2, normalized
    hyp1_seqs = [s2i("hello"), s2i("world")]
    truth1_seqs = [s2i("hallo"), s2i("would")]
    hyp1 = create_sparse_tensor(hyp1_seqs, dense_shape=(2, 5))
    truth1 = create_sparse_tensor(truth1_seqs, dense_shape=(2, 5))
    list_of_inputs.append(copy.deepcopy({
        'hypothesis': hyp1,
        'truth': truth1,
        'normalize': True,
        'name': 'rank2_normalized'
    }))

    # Input 2: Same as 1, but unnormalized
    list_of_inputs.append(copy.deepcopy({
        'hypothesis': hyp1,
        'truth': truth1,
        'normalize': False,
        'name': 'rank2_unnormalized'
    }))
    
    # Input 3: Perfect match
    hyp3_seqs = [s2i("test")]
    truth3_seqs = [s2i("test")]
    hyp3 = create_sparse_tensor(hyp3_seqs, dense_shape=(1, 4))
    truth3 = create_sparse_tensor(truth3_seqs, dense_shape=(1, 4))
    list_of_inputs.append(copy.deepcopy({
        'hypothesis': hyp3,
        'truth': truth3,
        'normalize': True,
        'name': 'perfect_match'
    }))

    # Input 4: Empty hypothesis sequences
    hyp4_seqs = [[], []]
    truth4_seqs = [s2i("abc"), s2i("de")]
    hyp4 = create_sparse_tensor(hyp4_seqs, dense_shape=(2, 5))
    truth4 = create_sparse_tensor(truth4_seqs, dense_shape=(2, 5))
    list_of_inputs.append(copy.deepcopy({
        'hypothesis': hyp4,
        'truth': truth4,
        'normalize': True,
        'name': 'empty_hypothesis'
    }))

    # Input 5: Empty truth sequences
    hyp5_seqs = [s2i("abc"), s2i("de")]
    truth5_seqs = [[], []]
    hyp5 = create_sparse_tensor(hyp5_seqs, dense_shape=(2, 5))
    truth5 = create_sparse_tensor(truth5_seqs, dense_shape=(2, 5))
    list_of_inputs.append(copy.deepcopy({
        'hypothesis': hyp5,
        'truth': truth5,
        'normalize': True,
        'name': 'empty_truth'
    }))

    # Input 6: Mixed empty and non-empty sequences
    hyp6_seqs = [s2i("ab"), []]
    truth6_seqs = [[], s2i("cd")]
    hyp6 = create_sparse_tensor(hyp6_seqs, dense_shape=(2, 5))
    truth6 = create_sparse_tensor(truth6_seqs, dense_shape=(2, 5))
    list_of_inputs.append(copy.deepcopy({
        'hypothesis': hyp6,
        'truth': truth6,
        'normalize': False,
        'name': 'mixed_empty'
    }))

    # Input 7: Rank-3 tensors, unnormalized
    hyp7_seqs = [[s2i("apple"), s2i("banana")], [s2i("test"), s2i("cat")]]
    truth7_seqs = [[s2i("apply"), s2i("bandana")], [s2i("text"), s2i("cot")]]
    hyp7 = create_sparse_tensor(hyp7_seqs, dense_shape=(2, 2, 7))
    truth7 = create_sparse_tensor(truth7_seqs, dense_shape=(2, 2, 7))
    list_of_inputs.append(copy.deepcopy({
        'hypothesis': hyp7,
        'truth': truth7,
        'normalize': False,
        'name': 'rank3_unnormalized'
    }))
    
    # Input 8: Pure integer sequences with a different numpy dtype
    hyp8_seqs = [[1, 2, 3], [10, 20]]
    truth8_seqs = [[1, 9, 3], [10, 30, 40]]
    hyp8 = create_sparse_tensor(hyp8_seqs, dense_shape=(2, 5), dtype=np.int32)
    truth8 = create_sparse_tensor(truth8_seqs, dense_shape=(2, 5), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({
        'hypothesis': hyp8,
        'truth': truth8,
        'normalize': True,
        'name': 'integer_sequences_int32'
    }))

    # Input 9: Complete mismatch
    hyp9_seqs = [s2i("abc")]
    truth9_seqs = [s2i("xyz")]
    hyp9 = create_sparse_tensor(hyp9_seqs, dense_shape=(1, 3))
    truth9 = create_sparse_tensor(truth9_seqs, dense_shape=(1, 3))
    list_of_inputs.append(copy.deepcopy({
        'hypothesis': hyp9,
        'truth': truth9,
        'normalize': True,
        'name': 'total_mismatch'
    }))

    # Input 10: Classic edit distance example with different lengths
    hyp10_seqs = [s2i("sitting"), s2i("cat")]
    truth10_seqs = [s2i("kitten"), s2i("cast")]
    hyp10 = create_sparse_tensor(hyp10_seqs, dense_shape=(2, 10))
    truth10 = create_sparse_tensor(truth10_seqs, dense_shape=(2, 10))
    list_of_inputs.append(copy.deepcopy({
        'hypothesis': hyp10,
        'truth': truth10,
        'normalize': False,
        'name': 'classic_example_unnormalized'
    }))

    return list_of_inputs

generated_inputs["tf.edit_distance"] = tf_edit_distance_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.edit_distance' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.edit_distance'.")

check_valid('tf.edit_distance', generated_inputs['tf.edit_distance'], lib="tf", suffix=0)
