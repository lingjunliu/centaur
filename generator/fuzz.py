import sys
from .harness import run_api_with_duration

def main():
    if len(sys.args) < 3:
        print("Usage: python fuzz.py <api> <duration>")
        return
    
    api = sys.args[1]
    duration = int(sys.args[2])
    
    run_api_with_duration(api, duration)

if __name__ == "__main__":
    main()