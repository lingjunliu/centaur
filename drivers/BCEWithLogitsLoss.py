import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Device configuration
    device = torch.device('cpu' if cpu else 'cuda')

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"]).to(device)
    target_tensor = torch.tensor(input["target"]).to(device)
    weight_tensor = torch.tensor(input.get("weight", None)).to(device) if input.get("weight", None) is not None else None
    reduction = input.get("reduction", 'mean')
    pos_weight_tensor = torch.tensor(input.get("pos_weight", None)).to(device) if input.get("pos_weight", None) is not None else None

    # Apply to torch.nn.BCEWithLogitsLoss
    criterion = torch.nn.BCEWithLogitsLoss(
        weight=weight_tensor,
        reduction=reduction,
        pos_weight=pos_weight_tensor
    )
    loss = criterion(input_tensor, target_tensor)

    return {"binary_cross_entropy_with_logits_loss": float(loss.item())}

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    device_string = "/cpu:0" if cpu else "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        target_tensor = tf.constant(input["target"])
        weight_tensor = tf.constant(input.get("weight", None)) if input.get("weight", None) is not None else None
        pos_weight_tensor = tf.constant(input.get("pos_weight", None)) if input.get("pos_weight", None) is not None else None

        # Apply to TensorFlow equivalent
        if pos_weight_tensor is not None:
            loss = tf.nn.weighted_cross_entropy_with_logits(
                labels=target_tensor, logits=input_tensor, pos_weight=pos_weight_tensor
            )
        else:
            loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=target_tensor, logits=input_tensor
            )

        if weight_tensor is not None:
            loss = loss * weight_tensor

        loss = tf.reduce_mean(loss)

        return {"binary_cross_entropy_with_logits_loss": float(loss.numpy())}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "target": np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float32),
        "weight": None,
        "reduction": 'mean',
        "pos_weight": None
    }

    # Torch example
    torch_result = torch_version(input_data, cpu=True)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data, cpu=True)
    print("TensorFlow result:", tf_result)

    # Comparison
    assert np.isclose(torch_result["binary_cross_entropy_with_logits_loss"], tf_result["binary_cross_entropy_with_logits_loss"], rtol=1e-5), "Results are not equal"
    
    if np.isclose(torch_result["binary_cross_entropy_with_logits_loss"], tf_result["binary_cross_entropy_with_logits_loss"], rtol=1e-5):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()