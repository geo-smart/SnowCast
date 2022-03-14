'''
The wrapper for all the snowcast_wormhole predictors
'''
class BaseHole:
  
  def __init__():
    self.classifier = get_model()
    self.train_x = None
    self.train_y = None
    self.test_x = None
    self.test_y = None
    
  
  def preprocessing():
    pass
  
  def train():
    self.classifier.fit(self.train_x, self.train_y)
  
  def test():
    self.test_y = self.classifier.predict(self.test_x)
  
  def validate():
    pass
  
  def predict():
    pass
  
  def get_model():
    pass
  
  def post_processing():
    pass
