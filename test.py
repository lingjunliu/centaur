from llm.valid_inputs_tf import generated_inputs
import random

selected_apis = random.sample(list(generated_inputs.keys()), 10)

print("Selected APIs:")
print('\n'.join(selected_apis))