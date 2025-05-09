import sys, os
import pandas as pd

def main():
    if len(sys.argv) < 2:
        print("Usage: python aggregate_csv.py <csv_file>")
        return
    csv_file = sys.argv[1]
    if not os.path.exists(csv_file):
        print(f"CSV file {csv_file} does not exist.")
        return
    # Read the CSV file
    df = pd.read_csv(csv_file)
    apis = df["api"].unique()
    # Initialize a dictionary to hold the aggregated results
    aggregated_results = {}
    for col in df.columns:
        aggregated_results[col] = []
    # Iterate through each unique API
    for api in apis:
        # Filter the DataFrame for the current API
        api_df = df[df["api"] == api]
        aggregated_results["api"].append(api)
        for col in api_df.columns:
            # Aggregate the results for each column
            if col == "api":
                continue
            elif col == "max_diff":
                # For max_diff, we want the maximum value
                aggregated_results[col].append(api_df[col].max())
            else:
                # For other columns, we want the sum
                aggregated_results[col].append(api_df[col].sum())
    
    # Create a new DataFrame from the aggregated results
    aggregated_df = pd.DataFrame(aggregated_results)
    # Save the aggregated DataFrame to a new CSV file
    output_file = os.path.splitext(csv_file)[0] + "_aggregated.csv"
    aggregated_df.to_csv(output_file, index=False)
    print(f"Aggregated results saved to {output_file}")
            
if __name__ == "__main__":
    main()