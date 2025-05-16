import numpy as np
import io

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    @torch.jit.script
    def my_module(x):
        return x

    module = my_module
    
    file = io.BytesIO()
    torch.jit.save(module, file)
    file.seek(0)
    flatbuffer = file.getvalue()
    
    flatbuffer_array = np.frombuffer(flatbuffer, dtype=np.uint8)

    return {"result": flatbuffer_array}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    import io

    @tf.function
    def func(x):
        return x

    concrete_func = func.get_concrete_function(tf.TensorSpec(shape=None, dtype=tf.float32))
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    tflite_model = converter.convert()
    
    tflite_array = np.frombuffer(tflite_model, dtype=np.uint8)
    
    return {"result": tflite_array}


def main():
    A_TOL = 1.0

    input_data = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    min_len = min(len(torch_result["result"]), len(tf_result["result"]))
    
    assert np.allclose(torch_result["result"][:min_len], tf_result["result"][:min_len], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()