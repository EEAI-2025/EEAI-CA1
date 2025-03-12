from model.randomforest import RandomForest
from model.chained_randomforest import ChainedRandomForest

"""
    CODE START
"""

def model_predict(data, df, name):
    results = []
    print("ChainedRandomForest")
    model = ChainedRandomForest("ChainedRandomForest", data.get_embeddings(), data.get_type())
    model.train(data)
    model.predict(data.X_test)
    model.print_results(data)
"""
CODE END
"""

def model_evaluate(model, data):
    model.print_results(data)