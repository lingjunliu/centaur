import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.linalg.inv_ex(input_tensor)
    
    if not cpu:
        result = (result[0].cpu(), result[1].cpu())
    
    return {"result": result[0].numpy(), "info": result[1].numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)

        try:
            result_matrix = tf.linalg.inv(input_tensor)
            ok = 0 
        except tf.errors.InvalidArgumentError:
            result_matrix = tf.linalg.pinv(input_tensor).numpy()
            result_matrix = tf.constant(result_matrix)
            ok = 1

        result_matrix = result_matrix.numpy()
        ok = np.array(ok)
    
    return {"result": result_matrix, "info": ok}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    assert np.allclose(torch_result["info"], tf_result["info"], atol=A_TOL), "Info results do not match"

    input_data = {
        "input": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    assert np.allclose(torch_result["info"], tf_result["info"], atol=A_TOL), "Info results do not match"

    input_data = {
        "input": np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    assert np.allclose(torch_result["info"], tf_result["info"], atol=A_TOL), "Info results do not match"

    input_data = {
        "input": np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    assert np.allclose(torch_result["info"], tf_result["info"], atol=A_TOL), "Info results do not match"

    input_data = {
        "input": np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    assert np.allclose(torch_result["info"], tf_result["info"], atol=A_TOL), "Info results do not match"


    print("Success")

if __name__ == "__main__":
    main()