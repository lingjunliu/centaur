from src.setseed import set_seed
import torch
import tensorflow as tf
import numpy as np

# Mockup of set_seed function; replace with actual implementation from src.setseed
def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    tensors = input["tensors"]
    tensors = [torch.tensor(t) for t in tensors]

    # Apply to torch.vstack
    result = torch.vstack(tensors)

    if not cpu:
        result = result.cpu()

    return {"vstack_result": result.numpy()}

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        tensors = input["tensors"]
        tensors = [tf.constant(t) for t in tensors]

        # Apply to TensorFlow equivalent (concatenate along first axis after ensuring 2D)
        tensors_2d = [tf.reshape(t, (1, -1) if tf.rank(t) == 1 else tf.shape(t)) for t in tensors]
        result = tf.concat(tensors_2d, axis=0)

        return {"vstack_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "tensors": [
            np.array([1, 2, 3], dtype=np.float32),
            np.array([4, 5, 6], dtype=np.float32)
        ]
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)
    
    # Compare results
    if np.array_equal(torch_result["vstack_result"], tf_result["vstack_result"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()