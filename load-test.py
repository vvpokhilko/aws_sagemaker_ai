import json  # Library for handling JSON data
import tarfile  # Library for creating tar archives

# Sample input data containing news headlines
input_data = [
    {"inputs": "Tesla Stock Bounces Back After Earnings"},
    {"inputs": "New Treatment for ADHD Discovered"},
    {"inputs": "Life Was Found on Planet Mars"},
    {"inputs": "Star Wars Remains Top Rated Movie"},
]

# Function to create JSON files from the input data
def create_json_files(data):
    for i, d in enumerate(data):  # Iterate over input data
        filename = f'input{i+1}.json'  # Generate filename dynamically
        with open(filename, 'w') as f:
            json.dump(d, f, indent=4)  # Write JSON data to file with indentation for readability

# Function to create a tar.gz archive containing the generated JSON files
def create_tar_file(input_files, output_filename='inputs.tar.gz'):
    with tarfile.open(output_filename, "w:gz") as tar:  # Open tar file in write mode
        for file in input_files:
            tar.add(file)  # Add each file to the archive

# Main function to execute the process
def main():
    create_json_files(input_data)  # Create JSON files from input data
    input_files = [f'input{i+1}.json' for i in range(len(input_data))]  # Generate list of filenames
    create_tar_file(input_files)  # Archive the JSON files into a tar.gz file

# Run the main function if the script is executed
def __name__ == '__main__':
    main()