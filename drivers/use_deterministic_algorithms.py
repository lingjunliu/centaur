from src.setseed import set_seed
import torch
import tensorflow as tf
import numpy as np

# Function to set seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Use deterministic algorithms in PyTorch
    torch.use_deterministic_algorithms(input["mode"])

    # For this example, we can create a simple calculation to demonstrate determinism
    data = torch.tensor(input["data"])
    
    if not cpu:
        data = data.cuda()

    result = torch.add(data, 1.0)  # Simple operation to demonstrate determinism

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
        tf.config.experimental.enable_op_determinism()
    else:
        device_string = "/gpu:0"
        tf.config.experimental.enable_op_determinism()

    with tf.device(device_string):
        # For this example, we can create a simple calculation to demonstrate determinism
        data = tf.constant(input["data"])
        result = tf.add(data, 1.0)  # Simple operation to demonstrate determinism

        return {"result": result.numpy()}

def main():
    # Example input
    input_data = {
        "mode": True,
        "data": np.array([0.5, 0.3, 0.8, 0.2, 0.6, 0.9], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert to check if results are equal
    assert np.array_equal(torch_result["result"], tf_result["result"]), "Results are not equal."
    
    print("equal")

if __name__ == "__main__":
    main()