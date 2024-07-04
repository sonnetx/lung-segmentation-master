import os
import shutil

def copy_files(file_list_path, source_directory, target_directory):
    # Ensure the target directory exists
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    
    with open(file_list_path, 'r') as file:
        file_names = file.read().splitlines()
    
    for file_name in file_names:
        source_path = os.path.join(source_directory, file_name)
        target_path = os.path.join(target_directory, file_name)
        
        if os.path.isfile(source_path):
            shutil.copy2(source_path, target_path)
            print(f'Copied {file_name} to {target_directory}')
        else:
            print(f'File {file_name} not found in {source_directory}')

# Specify the paths
file_list_path = '/home/groups/roxanad/sonnet/IMAGE_FILENAMES.txt'
source_directory = '/home/groups/roxanad/sonnet/mimic-cxr-jpg/2.1.0'
target_directory = '/home/groups/roxanad/sonnet/lung-segmentation-master/inputs'

# Copy the files
copy_files(file_list_path, source_directory, target_directory)
