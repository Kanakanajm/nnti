#!/bin/bash

# Check if a folder name is provided as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <folder_name>"
    exit 1
fi

# Assign the provided folder name to a variable
folder_name="$1"

# Check if the specified folder exists
if [ ! -d "$folder_name" ]; then
    echo "Error: Folder '$folder_name' not found."
    exit 1
fi

# Generate a new name for the zip file (you can customize this)
zip_name="HongluMa_7055053_CamiloMartinez_7057573.zip"

# Create a zip archive with the contents of the folder
zip -j $zip_name ./$folder_name/sol/*.pdf ./$folder_name/code/*.ipynb

# Check if the zip operation was successful
if [ $? -eq 0 ]; then
    echo "Zip archive '$zip_name' created successfully."
else
    echo "Error: Failed to create zip archive."
fi
