'''
The wrapper for all the snowcast_wormhole predictors
'''
class BaseHole:
  
  def __init__(self):
    self.classifier = self.get_model()
    self.train_x = None
    self.train_y = None
    self.test_x = None
    self.test_y = None
    
  
  def preprocessing(self):
    pass
  
  def train(self):
    self.classifier.fit(self.train_x, self.train_y)
  
  def test(self):
    self.test_y = self.classifier.predict(self.test_x)
  
  def validate(self):
    pass
  
  def predict(self):
    pass
  
  def get_model(self):
    pass
  
  def post_processing(self):
    pass
