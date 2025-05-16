import numpy as np


def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    if not cpu:
        sequences = [torch.tensor(seq).cuda() for seq in input["sequences"]]
    else:
        sequences = [torch.tensor(seq) for seq in input["sequences"]]

    batch_first = input.get("batch_first", False)
    padding_value = input.get("padding_value", 0.0)

    # Apply to torch.nn.utils.rnn.pad_sequence
    padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=batch_first, padding_value=padding_value)

    if not cpu:
        padded_sequences = padded_sequences.cpu()

    return {"padded_sequences": padded_sequences.numpy()}


def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        # Unpack input dictionary
        sequences = [tf.constant(seq) for seq in input["sequences"]]
        batch_first = input.get("batch_first", False)
        padding_value = input.get("padding_value", 0.0)
        
        # Get the max length for padding
        max_length = max(seq.shape[0] for seq in sequences)
        
        def pad_sequence(seq, maxlen, padding_value=0):
            pad_amount = maxlen - seq.shape[0]
            if pad_amount > 0:
                pad = tf.constant([padding_value] * pad_amount, dtype=seq.dtype)
                padded_seq = tf.concat([seq, pad], axis=0)
            else:
                padded_seq = seq
            return padded_seq
        
        padded_sequences = [pad_sequence(seq, max_length, padding_value) for seq in sequences]

        if batch_first:
            padded_sequences = tf.stack(padded_sequences, axis=0)
        else:
            padded_sequences = tf.stack(padded_sequences, axis=1)

        return {"padded_sequences": padded_sequences.numpy()}


def main():
    # Example input
    input_data = {
        "sequences": [
            [1.0, 2.0, 3.0],
            [4.0, 5.0],
            [6.0]
        ],
        "batch_first": False,
        "padding_value": 0.0
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    np.testing.assert_almost_equal(torch_result["padded_sequences"], tf_result["padded_sequences"])
    print("equal")


if __name__ == "__main__":
    main()