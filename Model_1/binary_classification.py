
from helper_methods import *

#Setup device
device= "cuda" if torch.cuda.is_available() else "cpu"

# Setup random seed
RANDOM_SEED = 42

X , y = make_moons(n_samples=1000,
                   random_state=42,   
                   noise = 0.07
                   )
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
# Turning data into tensors of dtype float
X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y, dtype=torch.float)


# Split the data into train and test sets (80% train, 20% test)

X_train,X_test,y_train,y_test = train_test_split(X,
                                                 y,
                                                 test_size=0.2,
                                                 random_state=42)


class MoonModelV0(nn.Module):
    def __init__(self, in_features, out_features, hidden_units):
      super().__init__()
      self.layers=nn.Sequential(
          nn.Linear(in_features=in_features,out_features=hidden_units),
          nn.ReLU(),
          nn.Linear(in_features=hidden_units,out_features=hidden_units),
          nn.ReLU(),
          nn.Linear(in_features=hidden_units,out_features=out_features),
          )


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.layers(x)


model_0 = MoonModelV0(in_features=2,
                      out_features=1,
                      hidden_units=10).to(device)


# Setup loss function
loss_fn = nn.BCEWithLogitsLoss()
# Setup optimizer to optimize model's parameters
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)

acc_fn = Accuracy().to(device)


epochs=1000

# Send data to the device
X_train,X_test,y_train,y_test = X_train.to(device),X_test.to(device),y_train.to(device),y_test.to(device)

# Loop through the data
for epoch in range(epochs):

  #Training

  model_0.train()

  # 1. Forward pass (logits output)
  y_logits =model_0(X_train.to(device)).squeeze()
  # logits -> prediction labels
  y_preds =torch.round(torch.sigmoid(y_logits))


  train_loss = loss_fn(y_logits, y_train) 

  train_accuracy= acc_fn(y_preds,y_train.int())

  optimizer.zero_grad()

  train_loss.backward()

  optimizer.step()

  ### Testing

  model_0.eval() 
  with torch.inference_mode():
    test_logits = model_0(X_test).squeeze()

    test_pred = torch.round(torch.sigmoid(test_logits))

    test_loss = loss_fn(test_logits, y_test)

    test_accuracy = acc_fn(test_pred, y_test.int()) 

  # Printing out results 
    if epoch % 10 == 0:
      print(f"Epoch: {epoch} | Loss: {train_loss:.2f} Acc: {train_accuracy:.2f} | Test loss: {test_loss:.2f} Test acc: {test_accuracy:.2f}")

#making predictions with the model
plot_decision_boundary(model_0,X_test,y_test)
