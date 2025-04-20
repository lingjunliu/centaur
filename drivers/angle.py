from src.setseed import set_seed
import torch
import tensorflow as tf
import numpy as np
import jax

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply torch.angle
    result = torch.angle(input_tensor)
    
    if not cpu:
        result = result.cpu()

    return {"angle": result.numpy()}

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

        # Apply TensorFlow equivalent (tf.math.angle)
        result = tf.math.angle(input_tensor)

        return {"angle": result.numpy()}

def jax_version(input, cpu=True):
    def angle_function(x): 
        return jax.numpy.angle(x)
    
    input_tensor = jax.numpy.array(input["input"])

    if cpu:
        device_string = "cpu"
    else:
        device_string = "gpu"

    device = jax.devices(device_string)[0] 
    angle = jax.jit(angle_function, backend=device_string) 
    result = angle(jax.device_put(input_tensor, device=device))
    return {"angle": np.array(result)}

def main():
    # Example input
    input_data = {
        "input": np.array([-1 + 1j, -2 + 2j, 3 - 3j], dtype=np.complex64)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Jax example
    jax_result = jax_version(input_data)
    print("Jax result:", jax_result)

    # Compare results
    torch_angle = np.array(torch_result["angle"])
    tf_angle = np.array(tf_result["angle"])
    jax_angle = np.array(jax_result["angle"])

    if np.allclose(torch_angle, tf_angle, atol=1e-5) and np.allclose(torch_angle, jax_angle, atol=1e-5):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()