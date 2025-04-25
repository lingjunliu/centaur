import sys
from .harness import run_api_with_duration

def main():
    if len(sys.argv) < 3:
        print("Usage: python fuzz.py <api> <duration> <n_max, optional>")
        return
    
    api = sys.argv[1]
    duration = int(sys.argv[2])
    n_max = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    
    run_api_with_duration(api, duration, n_max=n_max)

if __name__ == "__main__":
    main()