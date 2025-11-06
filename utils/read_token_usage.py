import sys, os

def read_token_usage(log_file):
    total_input_tokens = 0
    total_output_tokens = 0
    tokens_sum = 0
    if log_file.endswith('.log'):
        csv_file = log_file.replace('.log', '.csv')
    else:
        csv_file = log_file + '.csv'
    with open(csv_file, 'w') as f:
        f.write("input_tokens,output_tokens,total_tokens\n")

    with open(log_file, 'r') as f:
        lines = f.readlines()        
        for line in lines:
            # Match lines like this: Token usage: input=1476, output=5128, total=6604
            if line.strip().startswith("Token usage:"):
                parts = line.strip().split(',')
                input_tokens = int(parts[0].split('=')[1].strip())
                output_tokens = int(parts[1].split('=')[1].strip())
                total_tokens = int(parts[2].split('=')[1].strip())

                with open(csv_file, 'a') as csv_f:
                    csv_f.write(f"{input_tokens},{output_tokens},{total_tokens}\n")
        
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                tokens_sum += total_tokens

        with open(csv_file, 'a') as csv_f:
            csv_f.write(f"{total_input_tokens},{total_output_tokens},{tokens_sum}\n")

    print(f"Token usage data has been written to {csv_file}")
    return total_input_tokens, total_output_tokens, tokens_sum

def main():
    log_file = sys.argv[1] if len(sys.argv) > 1 else None

    if log_file is None:
        print("Usage: python read_token_usage.py <log_file>")
        sys.exit(0)
    elif os.path.isdir(log_file):
        dirs = os.listdir(log_file)
        sum_csv = os.path.join(log_file, 'token_usage_summary.csv')
        ultimate_inputs = 0
        ultimate_outputs = 0
        ultimate_totals = 0
        with open(sum_csv, 'w') as sum_f:
            sum_f.write("api,input_tokens,output_tokens,total_tokens\n")
        for d in dirs:
            cur_log_file = os.path.join(log_file, d, 'log-rulegen')
            if os.path.isfile(cur_log_file):
                input_tokens, output_tokens, total_tokens = read_token_usage(cur_log_file)
                with open(sum_csv, 'a') as sum_f:
                    sum_f.write(f"{d},{input_tokens},{output_tokens},{total_tokens}\n")
                ultimate_inputs += input_tokens
                ultimate_outputs += output_tokens
                ultimate_totals += total_tokens

        with open(sum_csv, 'a') as sum_f:
            sum_f.write(f"TOTAL,{ultimate_inputs},{ultimate_outputs},{ultimate_totals}\n")
    else:
        read_token_usage(log_file)

if __name__ == "__main__":
    main()