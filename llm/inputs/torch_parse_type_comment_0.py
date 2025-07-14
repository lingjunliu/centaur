
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def parse_type_comment_inputs():
    list_of_inputs = []

    # Input 1: Simple type comment
    input_dict = {"comment": "# type: (float) -> int"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Type comment with multiple arguments
    input_dict = {"comment": "# type: (int, str) -> bool"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Type comment with no return type
    input_dict = {"comment": "# type: (List[str])"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Type comment with Any type
    input_dict = {"comment": "# type: (Any) -> Any"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Type comment with complex type
    input_dict = {"comment": "# type: (Dict[str, List[int]]) -> Tuple[float, ...]" }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Type comment with Union type
    input_dict = {"comment": "# type: (Union[int, float]) -> str"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Type comment with Optional type
    input_dict = {"comment": "# type: (Optional[str]) -> None"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Type comment with Callable type
    input_dict = {"comment": "# type: (Callable[[int], str]) -> None"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Type comment with Tuple type
    input_dict = {"comment": "# type: (Tuple[int, int, int]) -> None"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Type comment with ellipsis in Tuple
    input_dict = {"comment": "# type: (Tuple[int, ...]) -> None"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.parse_type_comment"] = parse_type_comment_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.parse_type_comment' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.parse_type_comment'.")

check_valid('torch.parse_type_comment', generated_inputs['torch.parse_type_comment'], lib="torch", suffix=0)
