#!/bin/bash
# not working
# Specify the file containing the download links
input_file="/home/chetana/data/gridmet_test_run/amsr/download_links.txt"

# Specify the output directory for downloaded files
output_directory="/home/chetana/data/amsr_testing/amsr_curl_data/"

ls -l /home/chetana/data/gridmet_test_run/amsr/download_links.txt


while IFS= read -r url; do
    # Extract the filename from the URL
    filename=$(basename "$url")

    # Define the full path where the file will be saved
    output_path="$output_directory/$filename"

    echo "Downloading: $url -> $output_path"

    # Use curl to download the file with authentication cookies
    curl --output "$output_path" -b ~/.urs_cookies -c ~/.urs_cookies -L -n "$url"
done < "$input_file"
