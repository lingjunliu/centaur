import numpy as np
import pickle
import os
from io import BytesIO, BufferedReader

def torch_save_version(input, cpu=True):
    import torch

    obj = torch.tensor(input['obj'])
    f = input['f']
    pickle_module = input.get('pickle_module', pickle)
    pickle_protocol = input.get('pickle_protocol', pickle.HIGHEST_PROTOCOL)
    _use_new_zipfile_serialization = input.get('_use_new_zipfile_serialization', True)
    
    if not cpu:
        obj = obj.cuda()

    torch.save(obj, f, pickle_module=pickle_module, pickle_protocol=pickle_protocol, _use_new_zipfile_serialization=_use_new_zipfile_serialization)

    with open(f, 'rb') as file:
        saved_data = file.read()

    return {"torch_save_success": True, "data": saved_data}

def tensorflow_save_version(input, cpu=True):
    import tensorflow as tf

    obj = input['obj'].numpy() if isinstance(input['obj'], torch.Tensor) else input['obj']
    f = input['f']
    pickle_module = input.get('pickle_module', pickle)
    pickle_protocol = input.get('pickle_protocol', pickle.HIGHEST_PROTOCOL)

    with tf.io.gfile.GFile(f, 'wb') as file:
        pickle.dump(obj, file, protocol=pickle_protocol)

    with tf.io.gfile.GFile(f, 'rb') as file:
        saved_data = file.read()

    return {"tensorflow_save_success": True, "data": saved_data}

def load_saved_tensor(file_path, framework="torch"):
    if framework == "torch":
        return torch.load(file_path)
    elif framework == "tensorflow":
        with tf.io.gfile.GFile(file_path, 'rb') as file:
            return pickle.load(file)

def main():

    # Example input
    obj = torch.tensor([0, 1, 2, 3, 4])
    torch_file = 'torch_tensor.pt'
    tf_file = 'tf_tensor.pkl'

    input_data = {
        "obj": obj,
        "f": torch_file,
        "pickle_module": pickle,
        "pickle_protocol": pickle.HIGHEST_PROTOCOL,
        "_use_new_zipfile_serialization": True
    }

    # Torch example
    torch_result = torch_save_version(input_data)

    # TensorFlow example
    input_data_tf = input_data.copy()
    input_data_tf['f'] = tf_file
    tf_result = tensorflow_save_version(input_data_tf)

    # Load back the saved tensors
    torch_loaded_tensor = load_saved_tensor(torch_file, framework="torch")
    tf_loaded_tensor = load_saved_tensor(tf_file, framework="tensorflow")

    # Compare tensors
    if np.array_equal(torch_loaded_tensor.numpy(), tf_loaded_tensor):
        print("equal")
    else:
        print("not equal")

    # Clean up
    os.remove(torch_file)
    os.remove(tf_file)

if __name__ == "__main__":
    main()