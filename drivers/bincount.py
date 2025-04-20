import numpy as np

def torch_version(input, cpu=True):
    import torch
    
    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"], dtype=torch.int64)
    weights_tensor = torch.tensor(input.get("weights", None)) if input.get("weights", None) is not None else None
    minlength = input.get("minlength", 0)

    if not cpu:
        input_tensor = input_tensor.cuda()
        if weights_tensor is not None:
            weights_tensor = weights_tensor.cuda()
    
    # Apply to torch.bincount
    if weights_tensor is not None:
        output = torch.bincount(input_tensor, weights=weights_tensor, minlength=minlength)
    else:
        output = torch.bincount(input_tensor, minlength=minlength)
        
    if not cpu:
        output = output.cpu()

    return {"bincount_output": output.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"], dtype=tf.int64)
        weights_tensor = tf.constant(input.get("weights", None)) if input.get("weights", None) is not None else None
        minlength = input.get("minlength", 0)
        
        # Apply to TensorFlow equivalent
        if weights_tensor is not None:
            weighted_bincount = tf.math.bincount(input_tensor, weights=weights_tensor, minlength=minlength)
        else:
            weighted_bincount = tf.math.bincount(input_tensor, minlength=minlength)
        
        output = weighted_bincount.numpy()

    return {"bincount_output": output}

def main():
    # Example input
    input_data = {
        "input": np.array([4, 3, 6, 3, 4], dtype=np.int32),
        "weights": np.linspace(0, 1, 5, dtype=np.float32),
        "minlength": 0
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assertion
    torch_output = torch_result["bincount_output"]
    tf_output = tf_result["bincount_output"]

    if np.allclose(torch_output, tf_output, atol=1e-7):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()