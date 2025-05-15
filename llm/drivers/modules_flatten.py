import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    start_dim = input_dict.get("start_dim", 1)
    end_dim = input_dict.get("end_dim", -1)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.nn.Flatten(start_dim=start_dim, end_dim=end_dim)(input_tensor)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        start_dim = input_dict.get("start_dim", 1)
        end_dim = input_dict.get("end_dim", -1)
        
        shape = tf.shape(input_tensor)
        rank = tf.rank(input_tensor)
        
        if end_dim == -1:
            end_dim = rank - 1

        leading_dims = shape[:start_dim]
        middle_dims = shape[start_dim:end_dim+1]
        trailing_dims = shape[end_dim+1:]

        leading_size = tf.reduce_prod(leading_dims)
        middle_size = tf.reduce_prod(middle_dims)
        trailing_size = tf.reduce_prod(trailing_dims)

        if start_dim == 0 and end_dim == rank -1:
          new_shape = [-1]
        else:
          new_shape = tf.concat([tf.reshape(leading_size, [1]), tf.reshape(middle_size, [1]), tf.reshape(trailing_size, [1])], axis=0)
          
        if isinstance(new_shape, list):
            if len(new_shape) == 0:
                new_shape = [1]
        else:
            if tf.shape(new_shape)[0] == 0:
                new_shape = [1]
          
        result = tf.reshape(input_tensor, new_shape)
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(2, 3, 4).astype(np.float32),
        "start_dim": 1,
        "end_dim": 2
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"].flatten(), tf_result["result"].flatten(), atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(2, 3, 4).astype(np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"].flatten(), tf_result["result"].flatten(), atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(2, 3, 4).astype(np.float32),
        "start_dim": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"].flatten(), tf_result["result"].flatten(), atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.random.rand(2, 3, 4).astype(np.float32),
        "start_dim": 0,
        "end_dim": -1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"].flatten(), tf_result["result"].flatten(), atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()