import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import sys

    storage = input_dict["storage"]
    offset = input_dict.get("offset", 0)
    storage_offset = input_dict.get("storage_offset", 0)
    size = input_dict.get("size", len(storage))

    storage = torch.IntStorage.from_buffer(storage.tobytes(), byte_order=sys.byteorder)

    if not cpu:
        pass

    result = torch.IntStorage(size)
    for i in range(size):
        result[i] = storage[i+offset]
    
    if not cpu:
        pass
    
    return {"result": np.array(result)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    storage = input_dict["storage"]
    offset = input_dict.get("offset", 0)
    storage_offset = input_dict.get("storage_offset", 0)
    size = input_dict.get("size", len(storage))
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        result_list = []
        for i in range(size):
             result_list.append(storage[i+offset])

        result = np.array(result_list,dtype=np.int32)

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "storage": np.array([1, 2, 3, 4, 5], dtype=np.int32),
        "offset": 1,
        "size": 3
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()