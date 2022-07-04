'''
The wrapper for all the snowcast_wormhole predictors
'''
import os
from datetime import datetime

import joblib

homedir = os.path.expanduser('~')
github_dir = f"{homedir}/Documents/GitHub/SnowCast"


class BaseHole:

    def __init__(self):
        self.classifier = self.get_model()
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.test_y_results = None
        self.save_file = None

    def save(self):
        now = datetime.now()
        date_time = now.strftime("%Y%d%m%H%M%S")
        self.save_file = f"{github_dir}/model/wormhole_{date_time}.joblib"
        print(f"Saving model to {self.save_file}")
        joblib.dump(self.classifier, self.save_file)

    def preprocessing(self):
        pass

    def train(self):
        self.classifier.fit(self.train_x, self.train_y)

    def test(self):
        self.test_y_results = self.classifier.predict(self.test_x)
        return self.test_y_results

    def predict(self, input_x):
        return self.classifier.predict(input_x)

    def evaluate(self):
        pass

    def get_model(self):
        pass

    def post_processing(self):
        pass
