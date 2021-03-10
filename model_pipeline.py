import utils.helper_models as helper_models

class ModelPipeline:

    def __init__(self, data_read, data_explore, data_prepare,feature_engineering, modelling):
        self.data_read = data_read
        self.data_explore = data_explore
        self.data_prepare = data_prepare
        self.feature_engineering = feature_engineering
        self.modelling = modelling

    def fit(self, data_directory_path):

        print('Start model pipeline')

        try:

            data_raw = self.data_read.read(data_directory_path)
            df_clean = self.data_prepare.dataPrepare(data_raw)
            self.data_explore.dataExploration(df_clean)
            df_features_target = self.feature_engineering.featureEngineering(df_clean)
            self.modelling.modelling(df_features_target)
            df_features_target.to_csv('features_target.csv')

            print('Model Pipeline has been completed Successfully')

        except Exception as ex:
            print('Something went wrong with the Pipeline %s', ex)
            raise ex