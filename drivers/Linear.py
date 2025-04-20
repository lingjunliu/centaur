from src.setseed import set_seed
import torch
import tensorflow as tf
import numpy as np

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    tf.random.set_seed(seed)


def torch_version(input_dict, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    in_features = input_dict["in_features"]
    out_features = input_dict["out_features"]
    bias = input_dict.get("bias", True)
    input_tensor = torch.tensor(input_dict["input"])

    # Apply to torch.nn.Linear
    layer = torch.nn.Linear(in_features, out_features, bias=bias)
    if not cpu:
        layer = layer.cuda()
        input_tensor = input_tensor.cuda()
    
    output_tensor = layer(input_tensor)
    if not cpu:
        output_tensor = output_tensor.cpu()
        layer = layer.cpu()

    return {"output": output_tensor.detach().numpy(), "weight": layer.weight.detach().numpy(), "bias": layer.bias.detach().numpy() if bias else None}


def tensorflow_version(input_dict, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        in_features = input_dict["in_features"]
        out_features = input_dict["out_features"]
        bias_flag = input_dict.get("bias", True)
        input_tensor = tf.constant(input_dict["input"])
        weight = input_dict.get("weight", None)
        bias = input_dict.get("bias_tensor", None)

        # Initialize weights and biases with the same values as PyTorch
        if weight is None:
            k = np.sqrt(1.0 / in_features)
            weight = tf.Variable(tf.random.uniform([out_features, in_features], -k, k, seed=42), name="weight")
        else:
            weight = tf.Variable(weight.T, name="weight")  # Transpose to match TensorFlow's requirement
        
        if bias_flag:
            if bias is None:
                bias = tf.Variable(tf.random.uniform([out_features], -k, k, seed=42), name="bias")
            else:
                bias = tf.Variable(bias, name="bias")
        
        def linear_layer(x):
            y = tf.matmul(x, weight, transpose_b=False)  # No need to transpose again
            if bias_flag:
                y = tf.nn.bias_add(y, bias)
            return y

        output_tensor = linear_layer(input_tensor)
        
        return {"output": output_tensor.numpy()}


def main():
    # Example input
    input_data = {
        "in_features": 20,
        "out_features": 30,
        "bias": True,
        "input": np.random.randn(128, 20).astype(np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)
    
    input_data['weight'] = torch_result["weight"]
    input_data['bias_tensor'] = torch_result["bias"]

    # TensorFlow example with same weight and bias from PyTorch layer
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    is_equal = np.allclose(torch_result["output"], tf_result["output"], atol=1e-5)
    if is_equal:
        print("equal")
    else:
        print("not equal")


if __name__ == "__main__":
    main()