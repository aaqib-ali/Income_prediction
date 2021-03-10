import time
import argparse

# import from this project
from data_read import DataRead
from data_prepare import DataPrepare
from data_exploration import DataExploration
from feature_engineering import FeatureEngineering
from modelling import Modelling
from model_pipeline import ModelPipeline


def main(data_directory_path):

    print("Model Process starts")

    start = time.time()

    data_read = DataRead(data_directory_path)

    data_prepare = DataPrepare()

    data_explore = DataExploration()

    feature_engineering = FeatureEngineering()

    modelling = Modelling()

    model_pipeline = ModelPipeline(data_read, data_explore, data_prepare, feature_engineering, modelling)

    model_pipeline.fit(data_directory_path)

    print("Model Process ends", time.time() - start, "s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read the raw data')
    parser.add_argument('--data-file-path', type=str, default="./data",
                        help='Full path to raw data ')

    opt = parser.parse_args()
    print(opt)

    main(opt.data_file_path)



