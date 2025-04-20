import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    dim = input.get("dim", -1)
    descending = input.get("descending", False)
    stable = input.get("stable", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    # Apply to torch.argsort
    sorted_indices = torch.argsort(input_tensor, dim=dim, descending=descending, stable=stable)

    if not cpu:
        sorted_indices = sorted_indices.cpu()

    return {"argsort_indices": sorted_indices.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        axis = input.get("dim", -1)
        direction = 'DESCENDING' if input.get("descending", False) else 'ASCENDING'
        
        # TensorFlow's sorting doesn't have a `stable` argument directly equivalent
        # We will note this and assume the default behavior of TF are for stable = False
        sorted_indices = tf.argsort(input_tensor, axis=axis, direction=direction)
        
        return {"argsort_indices": sorted_indices.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "dim": 1,
        "descending": False,
        "stable": False
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Convert both outputs to numpy arrays and compare
    if np.allclose(torch_result["argsort_indices"], tf_result["argsort_indices"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()