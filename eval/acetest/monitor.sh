#!/bin/bash

# --- CONFIGURATION ---
# !!! IMPORTANT: SET THE PATH TO THE BASE DIRECTORY FOR COMPARISON !!!
# Example: If your subdirectories are './project_A', './project_B', etc., 
# and the comparison subdirectories are at '/backup/project_A', '/backup/project_B', 
# then COMPARISON_BASE_DIR should be '/backup'
COMPARISON_BASE_DIR="/workspace/out_tf_10m/output_tf_0"
# ---------------------

# Check if the comparison directory exists
if [ ! -d "$COMPARISON_BASE_DIR" ]; then
    echo "Error: Comparison base directory not found at $COMPARISON_BASE_DIR" >&2
    exit 1
fi

# Function to count files (excluding directories) in a given path
# Arguments: $1 = directory path
count_files_in_dir() {
    # -maxdepth 1 limits the search to the specified directory
    # -type f or -type l counts only files and symbolic links
    find "$1" -maxdepth 1 \( -type f -o -type l \) 2>/dev/null | wc -l
}

# Define the function to count and compare files
count_and_compare() {
    echo "Monitoring: $(pwd)"
    echo "Comparing against: $COMPARISON_BASE_DIR"
    echo "---"

    # Find all subdirectories (-mindepth 1 -type d) in the current directory (.), 
    # then process each one.
    find . -mindepth 1 -maxdepth 1 -type d -print -exec bash -c '
        CURRENT_DIR="$1"
        
        # Extract the subdirectory name (e.g., from "./Subfolder" -> "Subfolder")
        SUBDIR_NAME=$(basename "$CURRENT_DIR")
        
        # Calculate the path to the corresponding comparison directory
        COMP_DIR_PATH="'"$COMPARISON_BASE_DIR"'/""$SUBDIR_NAME"/non_crash
        
        # Count files in the current subdirectory
        CURRENT_COUNT=$(count_files_in_dir "$CURRENT_DIR")
        
        # Check if the comparison directory exists before counting
        if [ -d "$COMP_DIR_PATH" ]; then
            # Count files in the comparison subdirectory
            COMP_COUNT=$(count_files_in_dir "$COMP_DIR_PATH")
        else
            COMP_COUNT="N/A"
        fi
        
        # Print the result: [Directory Name]: [Current Count]/[Comparison Count]
        printf "%s: \t%s/%s files\n" "$SUBDIR_NAME" "$CURRENT_COUNT" "$COMP_COUNT"
    ' bash {} \; | sort
}

# Export the helper function and the variable so the inner find/exec bash can use them
export -f count_files_in_dir
export COMPARISON_BASE_DIR

# Use 'watch' to execute the 'count_and_compare' function every 1 second
count_and_compare