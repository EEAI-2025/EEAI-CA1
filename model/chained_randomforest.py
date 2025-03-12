import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.base import clone
from model.base import BaseModel

class ChainedRandomForest(BaseModel):
    def __init__(self,
                 model_name: str) -> None:
        super(ChainedRandomForest, self).__init__()
        self.model_name = model_name
        self.models = []

    def train(self, data_y2, data_y3, data_y4) -> None:
        # model_y2 = RandomForestClassifier(n_estimators=1000, random_state=42, class_weight='balanced_subsample')
        # model_y2.fit(data_y2.X_train, data_y2.y_train)  # Train y2
        # self.models.append(model_y2)

        # model_y3 = RandomForestClassifier(n_estimators=1000, random_state=42, class_weight='balanced_subsample')
        # model_y3.fit(data_y3.X_train, data_y3.y_train)  # Train y3
        # self.models.append(model_y3)

        # model_y4 = RandomForestClassifier(n_estimators=1000, random_state=42, class_weight='balanced_subsample')
        # model_y4.fit(data_y4.X_train, data_y4.y_train)  # Train y4
        # self.models.append(model_y4)


        model_y2 = RandomForestClassifier(n_estimators=1000, random_state=42, class_weight='balanced_subsample')
        print(f"\n🔹 Training RandomForest for y2 with data: {data_y2.X_train.shape} → {data_y2.y_train.shape}")
        print(f"  - Unique classes in y2:", np.unique(data_y2.y_train, return_counts=True))
        model_y2.fit(data_y2.X_train, data_y2.y_train)  # Train y2
        print(f"\n🔹 Training RandomForest for y2 with data: {data_y2.X_train.shape} → {data_y2.y_train.shape}")
        print(f"  - Unique classes in y2:", np.unique(data_y2.y_train, return_counts=True))
        self.models.append(model_y2)

        model_y3 = RandomForestClassifier(n_estimators=1000, random_state=42, class_weight='balanced_subsample')
        print(f"\n🔹 Training RandomForest for y3 with data: {data_y3.X_train.shape} → {data_y3.y_train.shape}")
        print(f"  - Unique classes in y3:", np.unique(data_y3.y_train, return_counts=True))
        model_y3.fit(data_y3.X_train, data_y3.y_train)  # Train y3
        print(f"\n🔹 Training RandomForest for y3 with data: {data_y3.X_train.shape} → {data_y3.y_train.shape}")
        print(f"  - Unique classes in y3:", np.unique(data_y3.y_train, return_counts=True))
        self.models.append(model_y3)

        model_y4 = RandomForestClassifier(n_estimators=1000, random_state=42, class_weight='balanced_subsample')
        print(f"\n🔹 Training RandomForest for y4 with data: {data_y4.X_train.shape} → {data_y4.y_train.shape}")
        print(f"  - Unique classes in y4:", np.unique(data_y4.y_train, return_counts=True))
        model_y4.fit(data_y4.X_train, data_y4.y_train)  # Train y4
        print(f"\n🔹 Training RandomForest for y4 with data: {data_y4.X_train.shape} → {data_y4.y_train.shape}")
        print(f"  - Unique classes in y4:", np.unique(data_y4.y_train, return_counts=True))
        self.models.append(model_y4)

    def predict(self, X_test: np.ndarray) -> None:
        predictions = np.zeros((X_test.shape[0], len(self.models)))

        for i, model in enumerate(self.models):
            if i > 0:
                X_test = np.column_stack((X_test, predictions[:, i - 1]))

            # Predict for the current output
            predictions[:, i] = model.predict(X_test)

        self.predictions = predictions

    def print_results(self,  data_y2, data_y3, data_y4) -> None:
        y_test = np.column_stack((data_y2.y_test, data_y3.y_test, data_y4.y_test))
        y_pred = self.predictions

        per_instance_accuracy = np.zeros(y_test.shape[0])
        num_labels = y_test.shape[1]

        for j in range(len(y_test)):
            if y_test[j, 0] != y_pred[j, 0]:  # If y2 is wrong, accuracy = 0%
                per_instance_accuracy[j] = 0.0
            elif y_test[j, 1] != y_pred[j, 1]:  # If y3 is wrong, accuracy = 33.33%
                per_instance_accuracy[j] = (1 / num_labels) * 100
            elif y_test[j, 2] != y_pred[j, 2]:  # If y4 is wrong, accuracy = 66.67%
                per_instance_accuracy[j] = (2 / num_labels) * 100
            else:  # All correct → 100%
                per_instance_accuracy[j] = 100.0

        print(per_instance_accuracy)
        overall_accuracy = np.mean(per_instance_accuracy)
        print("\n🔹 **Overall System Accuracy:**", f"{overall_accuracy:.2f}%")

    def data_transform(self) -> None:
        ...
