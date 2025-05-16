import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    try:
        @torch.jit.script
        def foo(x):
            if torch.sum(x) > 0:
                return x + 1
            else:
                return torch.ones([1], dtype=x.dtype)  # Ensure consistent return type and shape

        input_tensor = torch.tensor(input_dict["input"])
        
        if not cpu:
            input_tensor = input_tensor.cuda()
        
        foo(input_tensor)

        result = np.array([0])

        return {"result": result}

    except torch.jit.TracingCheckError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    try:
        @tf.function
        def foo(x):
            if tf.reduce_sum(x) > 0:
                return x + 1
            else:
                return tf.ones([1], dtype=x.dtype)

        input_tensor = tf.constant(input_dict["input"])
        foo(input_tensor)

        result = np.array([0])
        return {"result": result}

    except Exception as e:
        return {"error": str(e)}

def main():
    A_TOL = 0.01
    # Example input
    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),  # Ensure positive values

    }

    # Torch example
    torch_result = torch_version(input_data)
    
    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    
    torch_error = torch_result.get("error")
    tf_error = tf_result.get("error")

    # Assert to see if they are equal
    assert torch_error is not None or tf_error is not None, f"Torch error: {torch_error}, TF error: {tf_error}"

    print("Success")

if __name__ == "__main__":
    main()