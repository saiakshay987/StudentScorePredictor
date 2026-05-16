import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False)
            }
            params = {
                "Decision Tree": {
                    "max_depth": [3, 5, 7, 10]
                },
                "Random Forest": {
                    "n_estimators": [50, 100, 200],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.01, 0.1, 0.2],
                    "n_estimators": [50, 100, 200]
                },
                "K-Neighbors Regressor": {
                    "n_neighbors": [3, 5, 7, 10]
                },
                "XGBRegressor": {
                    "learning_rate": [0.01, 0.1, 0.2],
                    "n_estimators": [50, 100, 200]
                },
                "CatBoosting Regressor": {
                    "learning_rate": [0.01, 0.1, 0.2],
                    "n_estimators": [50, 100, 200]
                },
                "Linear Regression": {
                }
            }


            model_report = {}

            for i in range(len(models)):
                model = list(models.values())[i]
                gs = GridSearchCV(estimator=model, param_grid=params[list(models.keys())[i]], cv=5).fit(X_train, y_train)
                #model.fit(X_train, y_train)
                model.set_params(**gs.best_params_)
                model.fit(X_train, y_train)

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_model_score = r2_score(y_train, y_train_pred)
                test_model_score = r2_score(y_test, y_test_pred)

                model_report[list(models.keys())[i]] = test_model_score

            best_model_score = max(model_report.values())
            best_model_name = [k for k, v in model_report.items() if v == best_model_score][0]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                logging.info("No best model found")
                raise CustomException("No best model found", sys)

            logging.info(f"Best found model on both training and testing dataset: {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted) 

        except Exception as e:
            logging.info("Error occurred during model training")
            raise CustomException(e, sys)