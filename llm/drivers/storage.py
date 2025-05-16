import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

def torch_version(input_dict, cpu=True):
    input_tensor = torch.tensor(input_dict["input"])
    offset = int(input_dict["offset"])
    size = tuple(input_dict["size"])
    stride = tuple(input_dict["stride"])
    requires_grad = input_dict.get("requires_grad", False)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        
    result = input_tensor.as_strided(size, stride, storage_offset=offset)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    input_array = input_dict["input"]
    offset_val = input_dict["offset"]
    size_val = input_dict["size"]
    stride_val = input_dict["stride"]

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_array, dtype=tf.float32)
        offset = tf.constant(offset_val, dtype=tf.int32)
        size = tf.constant(size_val, dtype=tf.int32)
        stride = tf.constant(stride_val, dtype=tf.int32)
        
        input_shape = tf.shape(input_tensor)
        input_flat = tf.reshape(input_tensor, [-1])
        
        coords_gen = tf.meshgrid(*[tf.range(s) for s in size_val], indexing='ij')
        coords = tf.stack(coords_gen, axis=-1)

        strided_coords = coords * stride
        
        linear_ind = tf.reduce_sum(strided_coords, axis=-1) + offset
        
        linear_ind_flat = tf.reshape(linear_ind, [-1])

        gathered = tf.gather(input_flat, linear_ind_flat)
        
        output = tf.reshape(gathered, size)

    return {"result": output.numpy()}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
        "offset": 1,
        "size": (3,),
        "stride": (1,)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32),
        "offset": 0,
        "size": (2, 2),
        "stride": (1, 3)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    print("Success")

if __name__ == "__main__":
    main()