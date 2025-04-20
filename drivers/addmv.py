import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    mat_tensor = torch.tensor(input["mat"])
    vec_tensor = torch.tensor(input["vec"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        mat_tensor = mat_tensor.cuda()
        vec_tensor = vec_tensor.cuda()
    
    beta = input.get("beta", 1.0)
    alpha = input.get("alpha", 1.0)
    
    # Perform the operation
    result = torch.addmv(input=input_tensor, mat=mat_tensor, vec=vec_tensor, beta=beta, alpha=alpha)

    if not cpu:
        result = result.cpu()
    
    return {"addmv_result": result.cpu().numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        mat_tensor = tf.constant(input["mat"])
        vec_tensor = tf.constant(input["vec"])
        
        beta = input.get("beta", 1.0)
        alpha = input.get("alpha", 1.0)
        
        # Perform the operation
        mat_vec_product = tf.linalg.matvec(mat_tensor, vec_tensor)
        result = beta * input_tensor + alpha * mat_vec_product

        return {"addmv_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([0.5, 0.3], dtype=np.float32),
        "mat": np.array([[0.5, 0.2, 0.3], [0.4, 0.1, 0.6]], dtype=np.float32),
        "vec": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "beta": 1.0,
        "alpha": 1.0
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    torch_output = np.array(torch_result["addmv_result"])
    tf_output = np.array(tf_result["addmv_result"])

    if np.allclose(torch_output, tf_output, atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()