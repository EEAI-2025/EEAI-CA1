from controller.controller import Controller
from preprocessing.preprocess import Preprocessor
from controller.controller import  Controller
from utils.embeddings import *
from modelling.modelling import *
from modelling.data_model import *
from model.chained_randomforest import ChainedRandomForest

import random
seed =0
random.seed(seed)
np.random.seed(seed)

if __name__ == '__main__':
    # initial a controller object
    controller = Controller()

    # pre-processing steps
    df = controller.load_data()
    df = controller.preprocess_data(df)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')

    """
    CODE START
    """
    # y2 training embedding
    X, group_df = controller.get_embeddings(df)
    df_y2 = df.drop(columns=["y3", "y4"], axis=1)
    data_y2 = controller.get_data_object(X, df_y2, "y2")

    # y3 training embedding
    X, group_df = controller.get_embeddings(df, "y2")
    df_y3 = df.drop(columns=["y4"], axis=1)
    data_y3 = controller.get_data_object(X, df_y3, "y3")

    # y4 training embedding
    X, group_df = controller.get_embeddings(df, "y2", "y3")
    df_y4 = df.copy()
    data_y4 = controller.get_data_object(X, df_y4, "y4")

    model = ChainedRandomForest("ChainedRandomForest")
    model.train(data_y2, data_y3, data_y4)
    predictions = model.predict(data_y2.X_test)
    model.print_results(data_y2, data_y3, data_y4)

    """
    CODE END
    """