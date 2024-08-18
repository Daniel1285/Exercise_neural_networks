import os
import random
import shutil


base_dir = 'data'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


create_dir(test_dir)
create_dir(os.path.join(test_dir, 'cats'))
create_dir(os.path.join(test_dir, 'dogs'))

# Function to move 20% of images from train to test
def move_images(class_name):
    class_train_dir = os.path.join(train_dir, class_name)
    class_test_dir = os.path.join(test_dir, class_name)

    # Check if train directory exists
    if not os.path.exists(class_train_dir):
        print(f"Directory {class_train_dir} does not exist.")
        return

    # List all files in the class directory
    all_files = os.listdir(class_train_dir)

    # Calculate number of files to move (20% of total files)
    num_files_to_move = int(0.2 * len(all_files))

    # Randomly select files to move
    files_to_move = random.sample(all_files, num_files_to_move)

    # Move the selected files
    for file_name in files_to_move:
        src_path = os.path.join(class_train_dir, file_name)
        dest_path = os.path.join(class_test_dir, file_name)
        shutil.move(src_path, dest_path)
        print(f'Moved {file_name} to {class_test_dir}')

# Print paths for debugging
print(f"Training directory: {train_dir}")
print(f"Test directory: {test_dir}")

# Move images for each class
move_images('cats')
move_images('dogs')

print('Finished moving 20% of images from training to test directory.')
