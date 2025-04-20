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
    reduction = input.get("reduction", 'mean')

    # Compute the L1 loss using PyTorch
    loss_fn = torch.nn.L1Loss(reduction=reduction)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        target_tensor = target_tensor.cuda()
        loss_fn = loss_fn.cuda()

    loss = loss_fn(input_tensor, target_tensor)

    if not cpu:
        loss = loss.cpu()
        
    return {"L1Loss": float(loss.item())}

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
        reduction = input.get("reduction", 'mean')

        # Compute the L1 loss using TensorFlow
        if reduction == 'mean':
            loss = tf.reduce_mean(tf.abs(input_tensor - target_tensor))
        elif reduction == 'sum':
            loss = tf.reduce_sum(tf.abs(input_tensor - target_tensor))
        else:  # 'none'
            loss = tf.abs(input_tensor - target_tensor)
            return {"L1Loss": loss.numpy()}

        return {"L1Loss": float(loss.numpy())}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "target": np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float32),  # Ensure target is float type
        "reduction": 'mean'
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    assert np.isclose(torch_result["L1Loss"], tf_result["L1Loss"]), "The results are not equal!"
    print("equal")

if __name__ == "__main__":
    main()