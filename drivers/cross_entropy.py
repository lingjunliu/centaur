import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    target_tensor = torch.tensor(input["target"], dtype=torch.int64)  # Ensure target is long type for PyTorch cross_entropy
    weight_tensor = torch.tensor(input.get("weight", None)) if input.get("weight", None) is not None else None
    ignore_index = input.get("ignore_index", -100)
    reduction = input.get("reduction", 'mean')

    # Apply to torch.nn.functional.cross_entropy
    loss = torch.nn.functional.cross_entropy(
        input_tensor, target_tensor, weight=weight_tensor, ignore_index=ignore_index, reduction=reduction
    )

    if not cpu:
        loss = loss.cpu()

    return {"cross_entropy_loss": float(loss.item())}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        target_tensor = tf.constant(input["target"], dtype=tf.int32)  # Ensure target is int type for TensorFlow sparse_softmax_cross_entropy
        weight_tensor = tf.constant(input.get("weight", None)) if input.get("weight", None) is not None else None

        # Apply to TensorFlow equivalent
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=input_tensor, labels=target_tensor
        )

        if weight_tensor is not None:
            loss = loss * weight_tensor

        loss = tf.reduce_mean(loss)

        return {"cross_entropy_loss": float(loss.numpy())}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3]], dtype=np.float32),
        "target": np.array([0, 1], dtype=np.int64),  # Ensure target is int64 type for PyTorch cross_entropy
        "weight": None,
        "ignore_index": -100,
        "reduction": 'mean'
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assertion for comparison
    torch_loss = np.array([torch_result["cross_entropy_loss"]])
    tf_loss = np.array([tf_result["cross_entropy_loss"]])

    if np.isclose(torch_loss, tf_loss, atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()