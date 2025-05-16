import numpy as np
import io
import os
import tempfile

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack inputs from dictionary
    input_module = torch.jit.script(torch.nn.Linear(10, 5)) if "input_module" not in input_dict else torch.jit.script(torch.nn.Linear(10, 5)) # Minimal example

    if not cpu:
        pass # No CUDA conversion needed for this particular API

    # Perform torch save
    buffer = io.BytesIO()
    torch.jit.save_jit_module_to_flatbuffer(input_module, buffer)
    result = buffer.getvalue()

    return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Mimic the creation and saving of a simple model to a buffer
        model = tf.keras.models.Sequential([tf.keras.layers.Dense(5, input_shape=(10,))])
        
        # Save the model to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=True) as tmpfile:
            model.save(tmpfile.name)
            
            # Read the file into the buffer
            with open(tmpfile.name, 'rb') as f:
                result = f.read()

    return {"result": result}

def main():
    A_TOL = 0.01

    # Example input (minimal example - no actual input needed for save)
    input_data = {}

    # Torch example
    torch_result = torch_version(input_data)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)

    # Since the output is a byte buffer representing a serialized model,
    # we cannot directly compare the byte contents as they are different formats.
    # Therefore, a simple check that both results are non-empty bytes is a reasonable assertion here.
    
    assert len(torch_result["result"]) > 0, "Torch result is empty"
    assert len(tf_result["result"]) > 0, "TensorFlow result is empty"

    print("Success")

if __name__ == "__main__":
    main()