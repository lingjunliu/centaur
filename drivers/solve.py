import numpy as np

# Define the torch version of the function
def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    
    # Unpack input dictionary
    input_tensor = torch.tensor(input["B"])
    A_tensor = torch.tensor(input["A"])
    left = input.get("left", True)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        A_tensor = A_tensor.cuda()
    
    # Apply to torch.linalg.solve (or torch.solve for older versions)
    if hasattr(torch.linalg, 'solve'):
        result = torch.linalg.solve(A_tensor, input_tensor, left=left)
    else:
        result, _ = torch.solve(input_tensor, A_tensor, left=left)
    
    return {"solution": result.cpu().resolve_conj().numpy() if not cpu else result.resolve_conj().numpy()}

# Define the TensorFlow version of the function
def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["B"])
        A_tensor = tf.constant(input["A"])
        
        # TensorFlow solve doesn't support out parameter like PyTorch does
        result = tf.linalg.solve(A_tensor, input_tensor)
        
        return {"solution": result.numpy()}

# Main function that contains the example computations
def main():
    # Example input
    input_data = {
        "B": np.array([[2], [3]], dtype=np.float32),
        "A": np.array([[4, 2], [3, 1]], dtype=np.float32),
    }
    
    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)
    
    # Comparison
    if np.allclose(torch_result["solution"], tf_result["solution"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()