import pandas as pd
import os





class ReadWriteFiles:


    @staticmethod
    def read_file(file_path, class_index, with_headers=True):
        suffix = os.path.basename(file_path).split(".")[1]
        if suffix != 'csv':
            raise FileFormatError







class FileFormatError(Exception):
    def __str__(self):
        return "The Program Support Only CSV files."