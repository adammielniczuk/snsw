import os

def log_results(value, file_path):
    """
    Append the given value to the specified file. If the file does not exist, create it.

    Args:
        value (str): The value to be written to the file.
        file_path (str): The path to the file where the value should be logged.
    """
    try:
        complete_file_path = os.path.join(file_path, "results.txt")
        with open(complete_file_path, 'a') as file:
            file.write(value + '\n')
    except:
        print("Error occured while saving to a file")
