import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    padding = input_dict["padding"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    zero_pad = torch.nn.ZeroPad3d(padding)
    result = zero_pad(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    padding = input_dict["padding"]
    
    input_shape = input_tensor.shape
    rank = len(input_shape)
    
    if rank == 3:
        input_tensor = tf.reshape(input_tensor, (1, 1, input_shape[0], input_shape[1], input_shape[2]))
        rank = 5
    elif rank == 4:
        input_tensor = tf.reshape(input_tensor, (1, input_shape[0], input_shape[1], input_shape[2], input_shape[3]))
        rank = 5
    elif rank == 5:
        pass
    
    paddings = [[0, 0]] * rank
    
    if isinstance(padding, int):
        paddings[rank-3] = [padding, padding]
        paddings[rank-2] = [padding, padding]
        paddings[rank-1] = [padding, padding]
    elif len(padding) == 6:
        paddings[rank-3] = [padding[0], padding[1]]
        paddings[rank-2] = [padding[2], padding[3]]
        paddings[rank-1] = [padding[4], padding[5]]
    else:
        raise ValueError("Padding must be an int or a tuple of 6 ints.")
        
    result = tf.pad(input_tensor, paddings, "CONSTANT")
    
    if rank == 5:
        output_shape = [input_shape[i-2] + paddings[i][0] + paddings[i][1] for i in range(2,5)]
        result = tf.reshape(result, (1,1) + tuple(output_shape))
        result = tf.squeeze(result, axis=[0,1])
    elif rank == 4:
        output_shape = [input_shape[i-1] + paddings[i][0] + paddings[i][1] for i in range(1,5)]
        result = tf.reshape(result, (1,) + tuple(output_shape))
        result = tf.squeeze(result, axis=0)
    elif rank == 3:
        output_shape = [input_shape[i] + paddings[i+2][0] + paddings[i+2][1] for i in range(3)]
        result = tf.reshape(result, tuple(output_shape))



    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(2, 3, 4, 5, 6).astype(np.float32),
        "padding": (1, 1, 2, 2, 3, 3)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(3, 4, 5).astype(np.float32),
        "padding": 2
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()