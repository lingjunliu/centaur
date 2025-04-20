import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input1_tensor = torch.tensor(input["input1"])
    input2_tensor = torch.tensor(input["input2"])
    target_tensor = torch.tensor(input["target"])
    margin = input.get("margin", 0)
    reduction = input.get("reduction", 'mean')

    # Apply to torch.nn.functional.margin_ranking_loss
    loss = torch.nn.functional.margin_ranking_loss(
        input1=input1_tensor, input2=input2_tensor, target=target_tensor,
        margin=margin, reduction=reduction
    )

    if not cpu:
        loss = loss.cpu()

    return {"margin_ranking_loss": float(loss.item())}

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input1_tensor = tf.constant(input["input1"])
        input2_tensor = tf.constant(input["input2"])
        target_tensor = tf.constant(input["target"])
        margin = tf.constant(input.get("margin", 0))

        # Apply to TensorFlow equivalent function
        diff = input1_tensor - input2_tensor
        loss = tf.maximum(0., margin - target_tensor * diff)

        if input.get("reduction", 'mean') == 'mean':
            loss = tf.reduce_mean(loss)
        elif input.get("reduction") == 'sum':
            loss = tf.reduce_sum(loss)

        return {"margin_ranking_loss": float(loss.numpy())}

def main():
    # Example input
    input_data = {
        "input1": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "input2": np.array([3.0, 2.0, 1.0], dtype=np.float32),
        "target": np.array([1.0, -1.0, -1.0], dtype=np.float32),  # Ensure target is float type
        "margin": 0.5,
        "reduction": 'mean',
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Comparing results by converting them to the common format (numpy array of a single value, float)
    torch_loss = np.array([torch_result["margin_ranking_loss"]])
    tf_loss = np.array([tf_result["margin_ranking_loss"]])

    if np.allclose(torch_loss, tf_loss, atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()