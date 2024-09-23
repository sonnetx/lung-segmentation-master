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
        f_name = file_name.split(sep='/')[-1]
        target_path = os.path.join(target_directory, f_name)
        
        if os.path.isfile(source_path):
            if not os.path.exists(target_path):
                shutil.copy2(source_path, target_path)
                print(f'Copied {file_name} to {target_directory}')
                os.remove(source_path)
                print(f'Deleted original file {file_name} from {source_directory}')

            else:
                print(f'File {file_name} already exists in {target_directory}')
                os.remove(source_path)
                print(f'Deleted original file {file_name} from {source_directory}')

        else:
            print(f'File {file_name} not found in {source_directory}')

def remove_extra_files(file_list_path, target_directory):
    with open(file_list_path, 'r') as file:
        file_names = set(file.read().splitlines())
    
    for file_name in os.listdir(target_directory):
        if file_name not in file_names:
            file_path = os.path.join(target_directory, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f'Removed {file_name} from {target_directory}')
                
# Specify the paths
file_list_path = '/home/groups/roxanad/sonnet/lung-segmentation-master/test_set_jpgs.txt'
source_directory = '/home/groups/roxanad/sonnet/lung-segmentation-master/mimic-cxr-jpg/2.1.0'
target_directory = '/home/groups/roxanad/sonnet/lung-segmentation-master/inputs'

# Copy the files
copy_files(file_list_path, source_directory, target_directory)

# Remove extra files
remove_extra_files(file_list_path, target_directory)