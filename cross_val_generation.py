import numpy as np
import os
import pickle
# cross_val test


all_levels = ["Level1-1", "Level2-1", "Level3-1", "Level4-1", "Level5-1", "Level6-1", "Level7-1", "Level8-1"]

training_levels = []

for i in range(6):
    level_int = np.random.randint(len(all_levels))
    level = all_levels[level_int]
    all_levels.pop(level_int)
    training_levels.append(level)

print(training_levels)

os.makedirs("cross_validation_levels", exist_ok=True)

for j in range(10):
    with open(f"cross_validation_levels/index_{j}.pkl", "wb") as file:
        pickle.dump(training_levels, file)