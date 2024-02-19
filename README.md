# SFIIPrediction
SFIIPrediction is an open-source database designed for making binary winner predictions in the Street Fighter II video game. It operates by using data extracted from the percentage change in the health bar every 5 frames. To utilize the algorithm, open the file and execute it. You can modify the round_interv variable to any value between 0 and 1. This adjustment specifies the round progression percentage you wish to use as test data for making predictions.
```
# Set random seed for reproducibility
seed = 42
round_interv = 0.75
random.seed(a=seed)
np.random.seed(seed=seed)
torch.manual_seed(seed=seed)

# Set compute device to GPU if available or CPU otherwise
device = ("cuda" if torch.cuda.is_available() else "cpu")
```
