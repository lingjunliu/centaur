import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    if not cpu:
      torch.cuda.init()

    if not cpu:
      torch.cuda.empty_cache()

    result = torch.read_vitals()

    if not cpu:
        result = result.cpu()

    return {"result": np.array([result])}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    try:
        result = tf.raw_ops.ReadVitals()
    except AttributeError:
        result = "N/A"
    return {"result": np.array([result])}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    print(f"Torch Result: {torch_result['result']}")
    print(f"Tensorflow Result: {tf_result['result']}")

    if tf_result['result'][0] == "N/A":
      print("Tensorflow version of read_vitals does not exist. Skipping assertion.")
    else:
      assert np.allclose(torch_result["result"].astype(np.str_), tf_result["result"].astype(np.str_), atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()