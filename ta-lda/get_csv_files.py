
import os


def get_csv_files(root_dir):
    folders_and_files = {}
    for directories, sub_directories, files in os.walk(root_dir):
        # print(directories)
        # print(sub_directories)
        csv_files = [file for file in files if file.endswith('.csv')]
        if csv_files:
            sub_folder_name = os.path.relpath(directories, root_dir)
            # print(sub_folder_name)
            folders_and_files[sub_folder_name] = csv_files
    return folders_and_files


if __name__ == "__main__":

    root_directory = "./data/"
    csv_dict = get_csv_files(root_directory)
    # print(csv_dict.keys())
    # print(csv_dict.values())
    # first_key = next(iter(csv_dict))
    # print(first_key)
    # print(csv_dict[first_key])
