import os


class Config:
    def __init__(self, csv_file):
        self.csv_file = os.path.basename(csv_file)
        self.csv_file = os.path.splitext(self.csv_file)[0]

        self.cache_dir = 'cache'

        if not os.path.isdir(self.cache_dir):
            os.mkdir(self.cache_dir)

    @property
    def features_path(self):
        return os.path.join(self.cache_dir, os.path.splitext(self.csv_file)[0] + '_features')

    @property
    def model_path(self):
        return os.path.join(self.cache_dir, os.path.splitext(self.csv_file)[0] + '_model.h5')

    @property
    def history_path(self):
        return os.path.join(self.cache_dir, os.path.splitext(self.csv_file)[0] + '_history.txt')
