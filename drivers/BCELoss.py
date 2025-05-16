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

    # Apply sigmoid to convert logits to probabilities
    input_tensor = torch.sigmoid(input_tensor)

    # Apply torch.nn.BCELoss
    criterion = torch.nn.BCELoss(weight=weight_tensor, reduction=reduction)
    loss = criterion(input_tensor, target_tensor)

    if not cpu:
        loss = loss.cpu()

    return {"BCELoss": float(loss.item())}

# TensorFlow implementation for BCELoss equivalent
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
        target_tensor = tf.constant(input["target"])
        weight_tensor = tf.constant(input.get("weight", None)) if input.get("weight", None) is not None else None
        reduction = input.get("reduction", 'mean')

        # Apply sigmoid to convert logits to probabilities
        input_tensor = tf.sigmoid(input_tensor)

        # Apply TensorFlow equivalent of BCELoss
        loss = tf.keras.losses.binary_crossentropy(y_true=target_tensor, y_pred=input_tensor)

        if weight_tensor is not None:
            loss = loss * weight_tensor

        if reduction == 'mean':
            loss = tf.reduce_mean(loss)
        elif reduction == 'sum':
            loss = tf.reduce_sum(loss)

        return {"BCELoss": float(loss.numpy())}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "target": np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float32), # Ensure target is float type
        "weight": None,
        "reduction": 'mean',
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    if np.isclose(torch_result["BCELoss"], tf_result["BCELoss"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()