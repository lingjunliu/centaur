import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    target_tensor = torch.tensor(input["target"])
    weight_tensor = torch.tensor(input.get("weight", None)) if input.get("weight", None) is not None else None

    # Apply to torch.nn.functional.multilabel_soft_margin_loss
    loss = torch.nn.functional.multilabel_soft_margin_loss(
        input_tensor, target_tensor, weight=weight_tensor
    )

    if not cpu:
        loss = loss.cpu()

    return {"multilabel_soft_margin_loss": float(loss.item())}

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        target_tensor = tf.constant(input["target"])
        weight_tensor = tf.constant(input.get("weight", None)) if input.get("weight", None) is not None else None

        # Apply TensorFlow equivalent
        sigmoids = tf.nn.sigmoid(input_tensor)
        log_sigmoids = tf.math.log(sigmoids + 1e-16)
        log_one_minus_sigmoids = tf.math.log(1 - sigmoids + 1e-16)
        loss = -target_tensor * log_sigmoids - (1 - target_tensor) * log_one_minus_sigmoids

        if weight_tensor is not None:
            loss = loss * weight_tensor

        loss = tf.reduce_mean(loss)

        return {"multilabel_soft_margin_loss": float(loss.numpy())}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "target": np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float32),
        "weight": None
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    assert np.isclose(torch_result["multilabel_soft_margin_loss"], tf_result["multilabel_soft_margin_loss"]), "Results are not equal"
    print("equal")

if __name__ == "__main__":
    main()