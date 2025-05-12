import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.adjoint(input_tensor)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.resolve_conj().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        
        input_tensor_conj = tf.math.conj(input_tensor)
        
        rank = len(input_tensor.shape)
        if rank >= 2:
            perm = list(range(rank))
            perm[-1], perm[-2] = perm[-2], perm[-1]
            input_tensor_transposed = tf.transpose(input_tensor_conj, perm=perm)
        else:
            input_tensor_transposed = input_tensor_conj
            
        result = input_tensor_transposed
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[0. + 0.j, 1. + 1.j], [2. + 2.j, 3. + 3.j]], dtype=np.complex64)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()