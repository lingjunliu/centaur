import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input1_tensor = torch.tensor(input["x1"])
    input2_tensor = torch.tensor(input["x2"])
    p = input.get("p", 2.0)
    eps = input.get("eps", 1e-06)
    keepdim = input.get("keepdim", False)
    
    if not cpu:
        input1_tensor = input1_tensor.cuda()
        input2_tensor = input2_tensor.cuda()

    # Apply to torch.nn.PairwiseDistance
    pdist = torch.nn.PairwiseDistance(p=p, eps=eps, keepdim=keepdim)
    result = pdist(input1_tensor, input2_tensor)

    if not cpu:
        result = result.cpu()

    return {"pairwise_distance": result.numpy()}

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input1_tensor = tf.constant(input["x1"])
        input2_tensor = tf.constant(input["x2"])
        p = input.get("p", 2.0)
        eps = input.get("eps", 1e-06)
        keepdim = input.get("keepdim", False)

        # Apply to TensorFlow equivalent
        diff = input1_tensor - input2_tensor
        dist = tf.norm(diff + eps, ord=p, axis=-1, keepdims=keepdim)
        
        return {"pairwise_distance": dist.numpy()}

def main():
    # Example input
    input_data = {
        "x1": np.random.rand(100, 128).astype(np.float32),
        "x2": np.random.rand(100, 128).astype(np.float32),
        "p": 2.0,
        "eps": 1e-06,
        "keepdim": False
    }

    # Torch example
    torch_result = torch_version(input_data, cpu=True)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data, cpu=True)
    print("TensorFlow result:", tf_result)

    # Compare results in common format
    if np.allclose(torch_result["pairwise_distance"], tf_result["pairwise_distance"], atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()