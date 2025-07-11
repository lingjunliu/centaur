import sys

def main():
    log_file = sys.argv[1]

    read_valid = False
    read_total = False
    stat_dict = {}
    key = ""
    with open(log_file, 'r') as f:
        for line in f.readlines():
            if line.startswith("--- Generating"):
                # print(line.strip().split()[3])
                key = line.strip().split()[3]
                stat_dict[key] = []
                continue
            elif line.startswith(" -----"):
                read_valid = True
                continue
            if read_valid:
                # print(line.strip().split())
                read_valid = False
                read_total = True
                stat_dict[key].append(line.strip().split()[0])
            elif read_total:
                # print(line.strip().split())
                read_total = False
                stat_dict[key].append(line.strip().split()[0])
                continue
    
    print("api,valid,total,valid_prcnt")
    for k, v in stat_dict.items():
        if len(v) == 2:
            print(f"{k},{int(v[0])},{int(v[1])},{float(int(v[0])*100/int(v[1]) if int(v[0]) > 0 else 0):.2f}")
        else:
            print(f"{k},0,0,0.00")

if __name__ == "__main__":
    main()