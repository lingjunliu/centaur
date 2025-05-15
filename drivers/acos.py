import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply to torch.acos
    output_tensor = torch.acos(input_tensor)

    if not cpu:
        output_tensor = output_tensor.cpu()

    return {"acos_result": output_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])

        # Apply to TensorFlow equivalent tf.acos
        output_tensor = tf.acos(input_tensor)

        return {"acos_result": output_tensor.numpy()}
    
def jax_version(input, cpu=True):
    import jax
    def acos_function(x): 
        return jax.numpy.acos(x)
    
    input_tensor = jax.numpy.array(input["input"])

    if cpu:
        device_string = "cpu"
    else:
        device_string = "gpu"

    device = jax.devices(device_string)[0] 
    acos = jax.jit(acos_function, backend=device_string) 
    result = acos(jax.device_put(input_tensor, device=device))
    return {"acos_result": np.array(result)}

def main():
    # Example input
    input_data = {
        "input": np.array([0.3348, -0.5889, 0.2005, -0.1584], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    jax_result = jax_version(input_data)
    print("Jax result:", jax_result)

    # Compare results and print 'equal' or 'not equal'
    if np.allclose(torch_result["acos_result"], tf_result["acos_result"], atol=1e-6) and np.allclose(torch_result["acos_result"], jax_result["acos_result"], atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()