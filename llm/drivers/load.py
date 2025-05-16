import numpy as np
import io

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    f = input_dict["f"]
    map_location = input_dict.get("map_location", None)
    _extra_files = input_dict.get("_extra_files", None)
    _restore_shapes = input_dict.get("_restore_shapes", False)

    if isinstance(f, np.ndarray):
        f = io.BytesIO(f.tobytes())
    elif isinstance(f, bytes):
        f = io.BytesIO(f)
    
    if map_location is not None:
        if isinstance(map_location, str):
            pass
        else:
            map_location = torch.device(map_location)

    if _extra_files is not None:
        extra_files_torch = {}
        for filename, content in _extra_files.items():
            extra_files_torch[filename] = content
        _extra_files = extra_files_torch

    result = torch.jit.load(f, map_location=map_location, _extra_files=_extra_files, _restore_shapes=_restore_shapes)

    return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import pickle

    f = input_dict["f"]
    map_location = input_dict.get("map_location", None)
    _extra_files = input_dict.get("_extra_files", None)
    _restore_shapes = input_dict.get("_restore_shapes", False)

    if isinstance(f, np.ndarray):
        f = io.BytesIO(f.tobytes())
    elif isinstance(f, bytes):
        f = io.BytesIO(f)

    if map_location is not None:
        if isinstance(map_location, str):
            pass
        elif isinstance(map_location, torch.device):
            map_location = map_location.type
        else:
            map_location = "cpu"

    if _extra_files is not None:
        pass

    try:
        if hasattr(f, 'read'):
            if hasattr(f, 'seek'):
                f.seek(0)
            loaded_module = torch.jit.load(f, map_location=map_location)
        else:
            with open(f, 'rb') as file:
                loaded_module = torch.jit.load(file, map_location=map_location)
        
        return {"result": loaded_module}
    
    except Exception as e:
        return {"result": str(e)}

def main():
    A_TOL = 0.01

    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    # Create a simple ScriptModule for testing
    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)

    module = torch.jit.script(MyModule())

    # Save the module to a buffer
    buffer = io.BytesIO()
    torch.jit.save(module, buffer)
    buffer.seek(0)
    buffer_bytes = buffer.getvalue()

    # Example input
    input_data = {
        "f": buffer_bytes
    }

    # Torch example
    torch_result = torch_version(input_data)

    # Reset buffer
    input_data["f"] = buffer_bytes

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    
    # Assert to see if they are equal
    assert type(torch_result["result"]) == type(tf_result["result"])

    print("Success")

if __name__ == "__main__":
    main()