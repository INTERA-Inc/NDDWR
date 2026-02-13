#!/bin/bash

# Variables
INPUT_FILE="./clusterfiles2zip.txt"
TEMP_DIR="./temp_files"
ZIP_FILE="master_flow_00"

# Check if input file exists
if [[ ! -f $INPUT_FILE ]]; then
    echo "Error: $INPUT_FILE does not exist."
    exit 1
fi

# Clean up the input file to remove hidden characters, carriage returns, and trailing spaces
echo "Cleaning $INPUT_FILE..."
sed -i 's/\r//' "$INPUT_FILE"  # Removes Windows-style carriage returns
sed -i 's/[[:space:]]*$//' "$INPUT_FILE"  # Removes trailing spaces

# Create a temporary directory
if [[ -d $TEMP_DIR ]]; then
    echo "Removing existing temporary directory..."
    rm -rf $TEMP_DIR
fi
mkdir $TEMP_DIR

# Debug: Print the current working directory
echo "Current working directory: $(pwd)"

# Read each line from the input file and process it
while IFS= read -r line; do
    # Debug: Print the line being processed
    echo "Processing line: '$line'"

    # Expand wildcards in the line
    for file in $line; do
        if [[ -e "$file" ]]; then
            echo "Copying $file to $TEMP_DIR"
            cp -r "$file" "$TEMP_DIR"
        else
            echo "Warning: $file does not exist. Skipping."
        fi
    done
done < "$INPUT_FILE"

# Create a zip file from the temporary directory
echo "Creating tar.gz archive..."
tar -czf "$ZIP_FILE.tar.gz" -C "$TEMP_DIR" .


# Cleanup temporary directory
echo "Cleaning up temporary directory..."
rm -rf $TEMP_DIR

echo "Done. Created $ZIP_FILE."
