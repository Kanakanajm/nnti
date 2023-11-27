#!/bin/bash

# Function to display usage information
usage() {
  echo "Usage: $0 {pack|new} [options] [ASSIGNMENT FOLDER]"
  echo "  pack - Pack an assignment folder into a zip file."
  echo "    Options:"
  echo "      -o  Specify the output directory for the zip file."
  echo "  new  - Scaffold templates for new assignment (paper part)."
  exit 1
}

# Generate a new name for the zip file (you can customize this)
zip_name="HongluMa_7055053_CamiloMartinez_7057573.zip"

# Function to compress a folder into a zip file
pack_folder() {
  if [ -z "$folder_name" ]; then
    echo "Error: Please provide a folder name."
    exit 1
  fi

  output_dir=${output_dir:-.}  # If output directory is not provided, use the current directory
  zip_file="$output_dir/$zip_name"

  echo "Compressing $folder_name into $zip_file"
  # Create a zip archive with the contents of the folder
  zip -j "$zip_file" ./$folder_name/sol/*.pdf ./$folder_name/code/*.ipynb

  # Check if the zip operation was successful
  if [ $? -eq 0 ]; then
      echo "Zip archive '$zip_file' created successfully."
  else
      echo "Error: Failed to create zip archive."
  fi
}


# Function to create a new folder and copy 'template.tex' to 'sol'
create_new_folder() {
  if [ -z "$assignment_folder" ]; then
    echo "Error: Please provide an assignment folder name."
    exit 1
  fi

  folder_name="$assignment_folder"
  sol_folder="$folder_name/sol"

  echo "Creating new folder: $folder_name"
  mkdir -p "$sol_folder"
  cp template.tex "$sol_folder/"
  echo "Copied 'template.tex' to '$sol_folder/'"
}

# Main script starts here
if [ "$#" -lt 1 ]; then
  usage
fi

command=$1
shift

case $command in
  "pack")
    while getopts ":o:" opt; do
      case $opt in
        o)
          output_dir="$OPTARG"
          ;;
        \?)
          echo "Invalid option: -$OPTARG"
          usage
          ;;
        :)
          echo "Option -$OPTARG requires an argument."
          usage
          ;;
      esac
    done

    shift $((OPTIND-1))
    folder_name=$1
    pack_folder
    ;;

  "new")
    assignment_folder=$1
    create_new_folder
    ;;

  *)
    echo "Invalid command: $command"
    usage
    ;;
esac