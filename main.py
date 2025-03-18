from controller.controller import Controller
from controller.controller import  Controller
from utils.embeddings import *
from modelling.modelling import *
from modelling.data_model import *
from model.chained_randomforest import ChainedRandomForest

if __name__ == '__main__':
    # Initialize a controller object
    controller = Controller()

    # Preprocess and Clean the data
    df = controller.load_data()
    df = controller.preprocess_data(df)

    # Ensure all values in the column feature is a string value
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')

    # Get training and test data for type 2
    X, group_df = controller.get_embeddings(df)
    df_y2 = df.drop(columns=["y3", "y4"], axis=1)
    data_y2 = controller.get_data_object(X, df_y2, "y2")

    # Get training and test data for type 3
    X, group_df = controller.get_embeddings(df, "y2")
    df_y3 = df.drop(columns=["y4"], axis=1)
    data_y3 = controller.get_data_object(X, df_y3, "y3")

    # Get training and test data for type 4
    X, group_df = controller.get_embeddings(df, "y2", "y3")
    df_y4 = df.copy()
    data_y4 = controller.get_data_object(X, df_y4, "y4")

    # Train and predict for Chained Random Forest
    model = ChainedRandomForest("ChainedRandomForest")
    model.train(data_y2, data_y3, data_y4)
    predictions = model.predict(data_y2.X_test)
    model.print_results(data_y2, data_y3, data_y4)
