import os
import shutil
import random

def move_images(source_folder, target_folder, percentage):
    # Get all files in the source folder
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    total_files = len(files)
    
    # Calculate the number of files to move
    num_files_to_move = int(total_files * percentage)
    
    # Randomly select files to move
    files_to_move = random.sample(files, num_files_to_move)
    
    # Ensure the target folder exists
    os.makedirs(target_folder, exist_ok=True)
    
    # Move files
    for file_name in files_to_move:
        src_path = os.path.join(source_folder, file_name)
        dest_path = os.path.join(target_folder, file_name)
        shutil.move(src_path, dest_path)
    print(f"Moved {num_files_to_move} files from {source_folder} to {target_folder}.")

def main():
    # Define source and target directories
    base_dir = os.getcwd()  # Current working directory
    open_eyes_folder = os.path.join(base_dir, 'archive\mrleyedataset\Open-eyes')
    close_eyes_folder = os.path.join(base_dir, 'archive\mrleyedataset\Close-eyes')
    test_folder = os.path.join(base_dir, 'Data', 'test')

    # Move 5% of images
    move_images(open_eyes_folder, os.path.join(test_folder, 'Open-eyes'), 0.05)
    move_images(close_eyes_folder, os.path.join(test_folder, 'Close-eyes'), 0.05)

if __name__ == "__main__":
    main()

