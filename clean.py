import os

def delete_empty_folders(directory):
    """
    Deletes all empty folders in the specified directory.

    Args:
        directory (str): Path to the directory to search for empty folders.
    """
    if not os.path.exists(directory):
        print(f"The directory '{directory}' does not exist.")
        return

    # Walk the directory tree, bottom-up
    for root, dirs, files in os.walk(directory, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            # Check if the folder is empty
            if not os.listdir(dir_path):
                os.rmdir(dir_path)
                print(f"Deleted empty folder: {dir_path}")

if __name__ == "__main__":
    delete_empty_folders("checkpoints_06")
