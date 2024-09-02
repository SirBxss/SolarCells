import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import model

# Load the data from the CSV file and perform a train-test split
csv_path = 'data.csv'
tab = pd.read_csv(csv_path, sep=';')
train_tab, val_tab = train_test_split(tab, test_size=0.2, random_state=31)

# Set up data loading for the training and validation sets using DataLoader and ChallengeDataset
train_dl = t.utils.data.DataLoader(ChallengeDataset(train_tab, 'train'), batch_size=64, shuffle=True)
val_dl = t.utils.data.DataLoader(ChallengeDataset(val_tab, 'val'), batch_size=64)

# Create an instance of our ResNet model
model = model.ResNet()

# Set up a suitable loss criterion and optimizer, create a Trainer object, and set its early stopping criterion
crit = t.nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for binary classification
optimizer = t.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
trainer = Trainer(model, crit, optimizer, train_dl, val_dl, cuda=True, scheduler=scheduler)

# Call fit on the trainer
res = trainer.fit(epochs=50)

# Plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')
plt.show()
