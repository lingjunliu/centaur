import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"], dtype=torch.long)
    weight_tensor = torch.tensor(input["weight"])
    offsets = torch.tensor(input.get("offsets", None), dtype=torch.long) if input.get("offsets", None) is not None else None
    max_norm = input.get("max_norm", None)
    norm_type = input.get("norm_type", 2)
    scale_grad_by_freq = input.get("scale_grad_by_freq", False)
    mode = input.get("mode", 'mean')
    sparse = input.get("sparse", False)
    per_sample_weights = torch.tensor(input.get("per_sample_weights", None)) if input.get("per_sample_weights", None) is not None else None
    include_last_offset = input.get("include_last_offset", False)
    padding_idx = input.get("padding_idx", None)
    
    # Apply to torch.nn.functional.embedding_bag
    output = torch.nn.functional.embedding_bag(
        input_tensor, weight_tensor, offsets=offsets, max_norm=max_norm, norm_type=norm_type,
        scale_grad_by_freq=scale_grad_by_freq, mode=mode, sparse=sparse,
        per_sample_weights=per_sample_weights, include_last_offset=include_last_offset,
        padding_idx=padding_idx
    )

    if not cpu:
        output = output.cpu()

    return {"embedding_bag_output": output.detach().numpy()}

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"], dtype=tf.int32)
        weight_tensor = tf.constant(input["weight"])
        offsets = tf.constant(input.get("offsets", None), dtype=tf.int32) if input.get("offsets", None) is not None else None
        mode = input.get("mode", 'mean')

        # Perform the embedding lookup
        embedded = tf.nn.embedding_lookup(weight_tensor, input_tensor)

        # Aggregate embeddings according to the "mode" (mean, sum, max)
        if offsets is not None:
            # Calculate segments from offsets
            row_lengths = tf.concat([offsets[1:] - offsets[:-1], [tf.shape(input_tensor)[0] - offsets[-1]]], axis=0)
        else:
            row_lengths = tf.constant([embed.shape[0]], dtype=tf.int32)
        
        if mode == 'mean':
            output = tf.RaggedTensor.from_row_lengths(embedded, row_lengths).to_tensor()
            output = tf.reduce_mean(output, axis=1)
        elif mode == 'sum':
            output = tf.RaggedTensor.from_row_lengths(embedded, row_lengths).to_tensor()
            output = tf.reduce_sum(output, axis=1)
        elif mode == 'max':
            output = tf.RaggedTensor.from_row_lengths(embedded, row_lengths).to_tensor()
            output = tf.reduce_max(output, axis=1)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return {"embedding_bag_output": output.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([1, 2, 4, 5], dtype=np.int32),
        "weight": np.random.rand(6, 3).astype(np.float32),
        "offsets": np.array([0, 2], dtype=np.int32),
        "max_norm": None,
        "norm_type": 2,
        "scale_grad_by_freq": False,
        "mode": 'mean',
        "sparse": False,
        "per_sample_weights": None,
        "include_last_offset": False,
        "padding_idx": None
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Comparing results
    torch_output = torch_result["embedding_bag_output"]
    tf_output = tf_result["embedding_bag_output"]

    if np.allclose(torch_output, tf_output, atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()