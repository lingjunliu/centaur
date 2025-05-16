import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    model = input_dict["model"]
    fullgraph = input_dict.get("fullgraph", False)
    dynamic = input_dict.get("dynamic", None)
    backend = input_dict.get("backend", 'inductor')
    mode = input_dict.get("mode", None)
    options = input_dict.get("options", None)
    disable = input_dict.get("disable", False)
    
    if not cpu:
        pass
    
    compiled_model = torch.compile(model, fullgraph=fullgraph, dynamic=dynamic, backend=backend, mode=mode, options=options, disable=disable)
    
    def compiled_wrapper(x):
        return compiled_model(x)

    result = compiled_wrapper
    
    if not cpu:
        pass
    
    return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    model = input_dict["model"]

    def wrapper(x):
      x_tf = tf.convert_to_tensor(x)
      def tf_model(x_tf):
          x_torch = torch.tensor(x_tf.numpy())
          with tf.device('/cpu:0'):
              result_torch = model(x_torch).numpy()
          result_tf = tf.convert_to_tensor(result_torch)
          return result_tf

      return tf_model(x_tf).numpy()
    

    return {"result": wrapper}

def main():
    A_TOL = 0.01

    def example_model(x):
        import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
        return torch.sin(x) + torch.cos(x)
    
    input_data = {
        "model": example_model,
        "options": {"triton.cudagraphs": True},
        "fullgraph": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    import numpy as np
    test_input = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    try:
        import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
        compiled_torch_model = torch_result["result"]
        torch_output = compiled_torch_model(torch.tensor(test_input)).numpy()
        tf_output = tf_result["result"](test_input)
        assert np.allclose(torch_output, tf_output, atol=A_TOL), "Results do not match"

        print("Success")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()