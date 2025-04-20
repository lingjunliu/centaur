import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"], dtype=torch.long)
    weight_tensor = torch.tensor(input["weight"])
    padding_idx = input.get("padding_idx", None)
    max_norm = input.get("max_norm", None)
    norm_type = input.get("norm_type", 2.0)
    scale_grad_by_freq = input.get("scale_grad_by_freq", False)
    sparse = input.get("sparse", False)

    # Apply to torch.nn.functional.embedding
    output = torch.nn.functional.embedding(
        input_tensor,
        weight_tensor,
        padding_idx=padding_idx,
        max_norm=max_norm,
        norm_type=norm_type,
        scale_grad_by_freq=scale_grad_by_freq,
        sparse=sparse
    )

    if not cpu:
        output = output.cpu()

    return {"embedding": output.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"], dtype=tf.int64)
        weight_tensor = tf.constant(input["weight"])

        max_norm = input.get("max_norm", None)
        if max_norm is not None:
            weight_tensor = tf.clip_by_norm(weight_tensor, clip_norm=max_norm, axes=[1])

        output = tf.nn.embedding_lookup(params=weight_tensor, ids=input_tensor)

        if "padding_idx" in input and input["padding_idx"] is not None:
            padding_idx = input["padding_idx"]
            mask = tf.not_equal(input_tensor, padding_idx)
            output = tf.where(mask[:, :, tf.newaxis], output, tf.zeros_like(output))
        
        output = tf.identity(output) # Ensure computation is completed on the device
        
    return {"embedding": output.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=np.int64),
        "weight": np.random.rand(20, 3).astype(np.float32),
        "padding_idx": None,
        "max_norm": 1.0,
        "norm_type": 2.0,
        "scale_grad_by_freq": False,
        "sparse": False
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert equality
    torch_embedding = torch_result["embedding"]
    tf_embedding = tf_result["embedding"]
    
    if np.allclose(torch_embedding, tf_embedding, atol=1e-5):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()