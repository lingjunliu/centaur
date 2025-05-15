import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    # Unpack inputs from dictionary
    data_ptr = input_dict["data_ptr"]
    size = input_dict["size"][0]
    dtype = input_dict["dtype"]
    
    if not cpu:
        data_ptr = data_ptr.cuda()

    # Create a TypedStorage directly - not a typical use case but fulfilling API requirement.
    storage = torch.TypedStorage(dtype=dtype)
    storage.resize_(size)
    source = data_ptr.storage()
    storage.copy_(source)

    # Move result to CPU for consistent return format
    if not cpu:
        pass # storage does not have a .cpu() method, and the underlying data_ptr is already handled.
    
    return {"result": np.array([storage.size(), storage.nbytes(), storage.dtype().__str__()])}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    import numpy as np

    size = input_dict["size"]
    dtype_str = input_dict["dtype"].__str__()
    dtype = None

    if dtype_str == "torch.float32":
        dtype = tf.float32
    elif dtype_str == "torch.float64":
        dtype = tf.float64
    elif dtype_str == "torch.int64":
        dtype = tf.int64
    elif dtype_str == "torch.int32":
        dtype = tf.int32
    elif dtype_str == "torch.int8":
        dtype = tf.int8
    elif dtype_str == "torch.uint8":
        dtype = tf.uint8
    elif dtype_str == "torch.float16":
        dtype = tf.float16
    elif dtype_str == "torch.bool":
        dtype = tf.bool
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
        
    
    data_ptr_np = input_dict["data_ptr"].cpu().numpy()

    # Reconstruct the numpy array from the data pointer and size
    if data_ptr_np.size > 0:
        
        reshaped_array = data_ptr_np.reshape(size)

    else:
        reshaped_array = np.array([])

    # Simulate the storage properties
    storage_size = size[0] if isinstance(size, tuple) else size
    storage_nbytes = reshaped_array.nbytes
    storage_dtype_str = str(reshaped_array.dtype)

    return {"result": np.array([storage_size, storage_nbytes, dtype.__str__()])}



def main():
    import torch
    A_TOL = 0.01

    # Example input
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    data_ptr = torch.from_numpy(data)
    size = (4,)
    dtype = torch.float32

    input_data = {
        "data_ptr": data_ptr,
        "size": size,
        "dtype": dtype
    }

    # Torch example
    torch_result = torch_version(input_data)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)

    # Assert to see if they are equal
    try:
        assert np.allclose(torch_result["result"][:2].astype(float), tf_result["result"][:2].astype(float), atol=A_TOL), "Results do not match"
        assert torch_result["result"][2].split(".")[-1] in tf_result["result"][2] , "Dtypes do not match"
    except Exception as e:
        print(f"Torch Result: {torch_result['result']}")
        print(f"Tensorflow Result: {tf_result['result']}")
        raise e

    print("Success")

if __name__ == "__main__":
    main()