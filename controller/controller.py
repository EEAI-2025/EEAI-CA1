from preprocessing.preprocess import Preprocessor
from utils.embeddings import *
from modelling.modelling import *
from modelling.data_model import *

class Controller:
    def __init__(self):
        self.config = Config()
        self.preprocessor = Preprocessor()

    def load_data(self):
        # load the input data
        df = self.preprocessor.get_input_data()
        return df

    def preprocess_data(self,df):
        # De-duplicate input data
        df = self.preprocessor.de_duplication(df)
        # remove noise in input data
        df = self.preprocessor.noise_remover(df)
        # translate data to english
        # df[Config.TICKET_SUMMARY] = translate_to_en(df[Config.TICKET_SUMMARY].tolist())
        return df

    """
    CODE START
    """
    def get_embeddings(self, df: pd.DataFrame, *args):
        X = get_tfidf_embd(df, *args)
        return X, df

    def get_data_object(self, X: np.ndarray, df: pd.DataFrame, target_col: str):
        return Data(X, df, target_col)
    """
    CODE END
    """

    def perform_modelling(self,data: Data, df: pd.DataFrame, name):
        model_predict(data, df, name)