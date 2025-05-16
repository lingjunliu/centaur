import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    target_tensor = torch.tensor(input["target"])
    weight_tensor = torch.tensor(input.get("weight", None)) if input.get("weight", None) is not None else None
    reduction = input.get("reduction", 'mean')
    ignore_index = input.get("ignore_index", -100)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        target_tensor = target_tensor.cuda()
        if weight_tensor is not None:
            weight_tensor = weight_tensor.cuda()

    # Apply to torch.nn.functional.nll_loss
    loss = torch.nn.functional.nll_loss(
        input_tensor, target_tensor, weight=weight_tensor,
        reduction=reduction, ignore_index=ignore_index
    )

    if not cpu:
        loss = loss.cpu()

    return {"nll_loss": float(loss.item())}

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
        target_tensor = tf.constant(input["target"], dtype=tf.int64)
        weight_tensor = tf.constant(input.get("weight", None)) if input.get("weight", None) is not None else None
        ignore_index = input.get("ignore_index", -100)
        reduction = input.get("reduction", 'mean')

        # Create the one-hot encoding for the target tensor
        target_tensor_one_hot = tf.one_hot(target_tensor, depth=input_tensor.shape[-1])

        # Compute the actual negative log likelihood loss
        per_example_loss = -tf.reduce_sum(target_tensor_one_hot * input_tensor, axis=-1)

        # Apply weights if provided
        if weight_tensor is not None:
            per_example_loss *= weight_tensor

        # Apply ignore_index if provided
        if ignore_index is not None:
            target_mask = tf.not_equal(tf.cast(target_tensor, tf.int32), ignore_index)
            per_example_loss = tf.where(target_mask, per_example_loss, 0.0)

        # Apply reduction
        if reduction == 'mean':
            loss = tf.reduce_mean(per_example_loss)
        elif reduction == 'sum':
            loss = tf.reduce_sum(per_example_loss)
        else:
            loss = per_example_loss

        return {"nll_loss": float(loss.numpy())}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "target": np.array([2, 1], dtype=np.int64),  # Ensure target is int64 type
        "weight": None,
        "reduction": 'mean',
        "ignore_index": -100
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    if np.isclose(torch_result["nll_loss"], tf_result["nll_loss"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()