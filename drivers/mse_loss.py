from src.setseed import set_seed
import torch
import tensorflow as tf
import numpy as np

# PyTorch version function for mse_loss
def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()
    
    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    target_tensor = torch.tensor(input["target"])
    reduction = input.get("reduction", 'mean')
    size_average = input.get("size_average", True)
    reduce = input.get("reduce", True)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        target_tensor = target_tensor.cuda()
    
    # Apply to torch.nn.functional.mse_loss
    loss = torch.nn.functional.mse_loss(input_tensor, target_tensor, reduction=reduction, size_average=size_average, reduce=reduce)
    
    if not cpu:
        loss = loss.cpu()
    
    return {"mse_loss": float(loss.item())}

# TensorFlow version function for mse_loss
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
        
        # Apply to TensorFlow equivalent
        loss = tf.reduce_mean(tf.math.squared_difference(input_tensor, target_tensor))
        
        if reduction == 'sum':
            loss = tf.reduce_sum(loss)
        elif reduction == 'none':
            loss = tf.math.squared_difference(input_tensor, target_tensor)
        
        return {"mse_loss": float(loss.numpy())}

# Main function for testing
def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "target": np.array([[0.0, 0.1, 0.2], [0.2, 0.3, 0.4]], dtype=np.float32),  # Ensure target is float type
        "reduction": 'mean'
    }
    
    # Torch example
    torch_result = torch_version(input_data, cpu=True)
    print("Torch result:", torch_result)
    
    # TensorFlow example
    tf_result = tensorflow_version(input_data, cpu=True)
    print("TensorFlow result:", tf_result)
    
    # Comparison
    torch_loss = torch_result["mse_loss"]
    tf_loss = tf_result["mse_loss"]
    
    if np.isclose(torch_loss, tf_loss, atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()