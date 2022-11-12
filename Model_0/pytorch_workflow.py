
import torch 
import matplotlib.pyplot as plt
from torch import nn
torch.manual_seed(42) #setting a manual seed
weight = 0.7
bias= 0.3
X =torch.arange(0,1,0.02).unsqueeze(dim=1)
X.shape
y = weight * X + bias

#splitting the data for training and testing the model
train_split = int(0.8 * len(X)) # 80% of data used for training set, 20% for testing 
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]


def plot_predictions(train_data=X_train, 
                     train_labels=y_train, 
                     test_data=X_test, 
                     test_labels=y_test, 
                     predictions=None):
  
  plt.figure(figsize=(10, 7))

  # Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
  
  # Plot test data in green
  plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

  if predictions is not None:
    # Plot the predictions in red (predictions were made on the test data)
    plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

  # Show the legend
  plt.legend(prop={"size": 14});

#Predictions before training
plot_predictions()

#Creating a model

class LinearRegressionModel(nn.Module):

    def __init__(self):
      super().__init__()
      self.weight = nn.Parameter(torch.randn(1,requires_grad=True,dtype=torch.float)) #parameter with random value
      self.bias = nn.Parameter(torch.randn(1,requires_grad=True,dtype=torch.float)) #parameter with random value


    def forward(self, x:torch.Tensor) -> torch.Tensor : 
      return self.weight * x + self.bias


torch.manual_seed(42)

model_1 = LinearRegressionModel()
#Setting a loss function
loss_fn = nn.L1Loss()
#Setting an optimizer
optimizer = torch.optim.SGD(params = model_1.parameters(), lr=0.01)



#training loop 

epochs= 10000

for epoch in range(epochs):

  model_1.train()
  y_preds=model_1(X_train) #making predictions
  loss=loss_fn(y_preds,y_train)

  optimizer.zero_grad()

  loss.backward() #Backpropagation
  optimizer.step() #Gradient decent

  #testing

  model_1.eval()
  with torch.inference_mode():
    test_preds= model_1(X_test)
    loss_test = loss_fn(test_preds,y_test)
    if epoch%10000 == 0:
      print(loss_test)

plot_predictions(predictions=test_preds)