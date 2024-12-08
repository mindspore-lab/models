import os
import shutil


def reorganize_urban_sound_dataset(base_folder):
    # Create new fold directories
    for i in range(1, 11):
        fold_path = os.path.join(base_folder, f'new_fold{i}')
        os.makedirs(fold_path, exist_ok=True)

    # Traverse through all subdirectories and files
    for root, _, files in os.walk(base_folder):
        # Skip newly created fold directories
        if os.path.basename(root).startswith('new_fold'):
            continue

        for file in files:
            if file.endswith('.wav'):
                # Extract the label from the filename
                label = file.split('-')[1]
                # Determine the fold based on the label
                fold_number = (int(label) % 10) + 1
                # Destination path for the file
                dest_folder = os.path.join(base_folder, f'new_fold{fold_number}')
                dest_path = os.path.join(dest_folder, file)
                # Move the file to the new fold directory
                shutil.move(os.path.join(root, file), dest_path)

    # Remove all directories except the new fold directories
    for item in os.listdir(base_folder):
        item_path = os.path.join(base_folder, item)
        if os.path.isdir(item_path) and not item.startswith('new_fold'):
            shutil.rmtree(item_path)

    # Rename new_fold directories to fold
    for i in range(1, 11):
        new_fold_path = os.path.join(base_folder, f'new_fold{i}')
        fold_path = os.path.join(base_folder, f'fold{i}')
        os.rename(new_fold_path, fold_path)


# Example usage:
base_folder = 'UrbanSound8K'
reorganize_urban_sound_dataset(base_folder)