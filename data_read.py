import pandas as pd

class DataRead:

    def __init__(self, directory_path):
        self.directory_path = directory_path

    # We have only one file of data, in case of multiple files logic of read will be changed accordingly.
    def read(self, directory_path):

        print('Start Reading Raw data from:', directory_path)
        try:

            pd.options.display.max_columns = 50
            pd.options.display.max_rows = 50

            data_raw = pd.read_csv(directory_path)
            #print(data_raw.head(100))

            print('Successfully read Raw date...')

            return data_raw

        except Exception as ex:
            print('File can not be read%s', ex)
            raise ex