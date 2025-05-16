import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()

    schema_string = input_dict["schema"]
    result = torch.parse_schema(schema_string)

    if not cpu:
        result = result.cpu()

    return {"result": str(result)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.convert_to_tensor(input_dict["input"])
    schema_string = input_dict["schema"]

    result = schema_string

    return {"result": str(result)}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1, 2, 3, 4, 5, 6], dtype=np.int64),
        "schema": "aten::add(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert torch_result["result"] == tf_result["result"], "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()