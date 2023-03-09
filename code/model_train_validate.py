from model_creation_rf import RandomForestHole
from model_creation_xgboost import XGBoostHole

print("Train Models")

worm_holes = [RandomForestHole(), XGBoostHole()]

for hole in worm_holes:
  hole.preprocessing()
  print("Training_X model shape:",hole.train_x.shape)
  print("Training_Y model shape:",hole.train_y.shape)
  hole.train()
  hole.test()
  hole.evaluate()
  hole.save()
  
print("Finished training and validating all the models.")

