import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    target_tensor = torch.tensor(input["target"], dtype=torch.long)  # Ensure target is long type for class indices
    weight_tensor = torch.tensor(input.get("weight", None)) if input.get("weight", None) is not None else None
    reduction = input.get("reduction", 'mean')
    ignore_index = input.get("ignore_index", -100)
    label_smoothing = input.get("label_smoothing", 0.0)

    # Apply to torch.nn.CrossEntropyLoss
    criterion = torch.nn.CrossEntropyLoss(
        weight=weight_tensor,
        reduction=reduction,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing
    )
    loss = criterion(input_tensor, target_tensor)

    if not cpu:
        loss = loss.cpu()

    return {"cross_entropy_loss": float(loss.item())}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        target_tensor = tf.constant(input["target"], dtype=tf.int32)  # Ensure target is int type for class indices
        weight_tensor = tf.constant(input.get("weight", None)) if input.get("weight", None) is not None else None
        label_smoothing = input.get("label_smoothing", 0.0)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_tensor, logits=input_tensor
        )

        if weight_tensor is not None:
            target_one_hot = tf.one_hot(target_tensor, depth=input_tensor.shape[-1])
            sample_weights = tf.reduce_sum(target_one_hot * weight_tensor, axis=-1)
            loss = loss * sample_weights

        loss = tf.reduce_mean(loss)

        return {"cross_entropy_loss": float(loss.numpy())}

def main():
    # Example input
    input_data = {
        "input": np.random.randn(3, 5).astype(np.float32),
        "target": np.random.randint(0, 5, size=(3,)).astype(np.int32),  # Ensure target is int type for class indices
        "weight": np.ones(5, dtype=np.float32),
        "reduction": 'mean',
        "ignore_index": -100,
        "label_smoothing": 0.0
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert for comparison, we use numpy.allclose to handle small differences in floating point precision
    if np.allclose(torch_result["cross_entropy_loss"], tf_result["cross_entropy_loss"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()