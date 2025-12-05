# Preparation Script

import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset from local file
data = pd.read_csv("dataset.csv")

# Random split into learning and test sets
learning_df, test_df = train_test_split(
    data,
    test_size=0.2,       # 80% learning, 20% test
    random_state=0       # fixed seed for reproducibility
)

# Save the two sets as Python objects in the working directory
learning_df.to_pickle("learning.pkl")
test_df.to_pickle("test.pkl")

# Small check
print("Learning set shape:", learning_df.shape)
print("Test set shape:", test_df.shape)
