from generator.rules import check_rules
from .inputs import get_inputs
from utils.api_utils import get_driver

def infer_invariants(api, list_of_inputs, print_details=False):
    initialized = False
    ruleset = set()
    print(f"Inferring invariants for {api} with {len(list_of_inputs)} inputs\n")
    for idx, input_dict in enumerate(list_of_inputs):
        try:
            out_cpu = get_driver(api)(input_dict, cpu=True)
            if print_details:
                print(f"Input {idx} is valid")
            # Check rules for the input dictionary
            if not initialized:  # If ruleset is not initialized
                ruleset = check_rules(input_dict)
                initialized = True
            else:
                ruleset = ruleset.intersection(check_rules(input_dict))
        except:
            if print_details:
                print(f"Input {idx} is invalid")   
    
    return ruleset

def save_invariants_as_csv(api, ruleset):
    # Print the rules that have been passed for the scatter operation
    if len(ruleset) > 0:
        invariant_file = f"invariants/{api}.csv"
        # If there are rules that have been passed, print them
        with open(invariant_file, "w") as fi:
            fi.write("api,rule,arg1,arg2\n")
            print(f"Rules passed for {api}:")
            for rule, arg1, arg2 in ruleset:
                print(f"- {rule} between {arg1} and {arg2}")
                fi.write(f"{api},{rule},{arg1},{arg2}\n")
    else:
        print(f"No rules passed for {api}.")

def main():
    # scatter
    api = "scatter"
    ruleset = infer_invariants(api, get_inputs(api))
    save_invariants_as_csv(api, ruleset)
    
    print("\n-----\n")
    
    # conv_transpose2d
    api = "conv_transpose2d"
    ruleset = infer_invariants(api, get_inputs(api))
    save_invariants_as_csv(api, ruleset)

if __name__ == "__main__":
    main()