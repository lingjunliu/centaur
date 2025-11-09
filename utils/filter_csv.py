import sys

def main():
    csv_file = sys.argv[1]
    api_list = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else csv_file.replace('.csv', '_filtered.csv')

    with open(api_list, 'r') as api_f:
        api_set = set([line.strip() for line in api_f.readlines()])

    with open(csv_file, 'r') as f, open(output_file, 'w') as out_f:
        for line in f:
            tokens = line.strip().split(',')
            if tokens[0] in api_set:
                out_f.write(line)

if __name__ == "__main__":
    main()