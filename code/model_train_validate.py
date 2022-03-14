from model_creation_rf import RandomForestHole

print("Train Models")

worm_holes = [RandomForestHole()]

for hole in worm_holes:
  hole.train()
  hole.validate()
  hole.save()
  
print("Finished training and validating all the models.")

