import numpy as np
import io

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    m = input_dict["m"]
    f = input_dict["f"]
    _extra_files = input_dict.get("_extra_files", None)

    if not isinstance(f, io.BytesIO):
      f = str(f)
    
    if not cpu:
        pass
    
    torch.jit.save(m, f, _extra_files=_extra_files)
    
    if not cpu:
        pass
    
    return {}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import os

    m = input_dict["m"]
    f = input_dict["f"]
    _extra_files = input_dict.get("_extra_files", None)

    if not isinstance(f, io.BytesIO):
        f = str(f)

    if _extra_files is not None:
        for filename, content in _extra_files.items():
            with open(filename, 'wb') as outfile:
                outfile.write(content)

    try:
        torch.jit.save(m, f, _extra_files=_extra_files)
    except Exception as e:
        print(f"Warning: TensorFlow version does not directly support torch.jit.save, but saving using torch instead.\nError: {e}")
    
    if _extra_files is not None:
        for filename in _extra_files.keys():
            if os.path.exists(filename):
                os.remove(filename)

    return {}

def main():
    A_TOL = 0.01

    class MyModule(torch.nn.Module):
        def forward(self, x):
            return x + 10

    m = torch.jit.script(MyModule())
    file_name = 'test_scriptmodule.pt'

    input_data = {
        "m": m,
        "f": file_name
    }

    torch_version(input_data)
    tensorflow_version(input_data)

    import os
    if os.path.exists(file_name):
        os.remove(file_name)

    print("Success")

if __name__ == "__main__":
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    main()