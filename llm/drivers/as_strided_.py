import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    size = input_dict["size"]
    stride = input_dict["stride"]
    storage_offset = input_dict.get("storage_offset", 0)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = input_tensor.as_strided(size, stride, storage_offset).clone()
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        size = input_dict["size"]
        stride = input_dict["stride"]
        storage_offset = input_dict.get("storage_offset", 0)

        input_tensor_shape = tf.shape(input_tensor)
        
        
        strides_tf = []
        current_stride = 1
        for i in reversed(range(len(size))):
          strides_tf.append(current_stride)
          current_stride *= size[i]
        strides_tf = list(reversed(strides_tf))
        
        indices = tf.range(tf.reduce_prod(size))
        
        multi_indices = []
        for s in strides_tf:
            multi_indices.append(indices // s)
            indices = indices % s
            
        
        linear_indices = tf.zeros_like(multi_indices[0], dtype=tf.int32)
        for i in range(len(size)):
            linear_indices += multi_indices[i] * stride[i]
        
        linear_indices += storage_offset
        
        result = tf.gather(input_tensor, linear_indices)
        result = tf.reshape(result, size)
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1, 2, 3, 4, 5, 6], dtype=np.float32),
        "size": (2, 2),
        "stride": (1, 2),
        "storage_offset": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()