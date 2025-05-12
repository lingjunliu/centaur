import numpy as np
import os

def torch_version(input_dict, cpu=True):
    import torch

    filename = input_dict["filename"]
    _extra_files = input_dict.get("_extra_files", {})
    
    class MyModule(torch.nn.Module):
        def forward(self, x):
            return x + 1

    module = torch.jit.script(MyModule())

    if not cpu:
        pass
    
    module.save(filename, _extra_files=_extra_files)
    
    if not cpu:
        pass
    
    return {}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    filename = input_dict["filename"]
    _extra_files = input_dict.get("_extra_files", {})
    filename = filename.replace(".pt", "_tf") 
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        def dummy_function(x):
            return x + 1.0

        concrete_function = tf.function(dummy_function).get_concrete_function(tf.TensorSpec(shape=None, dtype=tf.float32))
        
        tf.saved_model.save(
            obj=concrete_function,
            export_dir=filename
        )

        if _extra_files:
            os.makedirs(filename, exist_ok=True)
            for k, v in _extra_files.items():
                with open(os.path.join(filename, k), "w") as f:
                    f.write(v)

    return {}

def main():
    A_TOL = 0.01
    input_data = {
        "filename": "test_model.pt",
        "_extra_files": {"extra.txt": "This is extra content."}
    }

    torch_version(input_data)
    tensorflow_version(input_data)

    print("Success")

if __name__ == "__main__":
    main()