def log_results(value, file_path):
    """
    Append the given value to the specified file. If the file does not exist, create it.

    Args:
        value (str): The value to be written to the file.
        file_path (str): The path to the file where the value should be logged.
    """
    try:
        with open(file_path, 'a') as file:
            file.write(value + '\n')
    except Exception as e:
        print(f"An error occurred while logging to the file: {e}")
