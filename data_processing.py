import json   
import numpy as np
from sklearn.model_selection import train_test_split
import torch  
from torch.utils.data import DataLoader, TensorDataset
import pdb



def process_data(geste: str, max_len: int, len_train, len_test):
    json_file = open(f"raw_{geste}_data.json", 'r')
    list_of_lists = json.load(json_file)
    json_file.close()
    list_of_lists = pad_or_truncate(list_of_lists, max_len)
    print(len(list_of_lists))
    train_data = np.array(list_of_lists[:len_train])
    test_data = np.array(list_of_lists[len_train:len_train+len_test])
    np.save(f"{geste}_train_data.npy", train_data)
    np.save(f"{geste}_test_data.npy", test_data)

def pad_or_truncate(list_of_lists: list, max_len: int):
    for i in range(len(list_of_lists)):
        if len(list_of_lists[i]) > max_len:
            list_of_lists[i] = list_of_lists[i][:max_len]
        else:
            while len(list_of_lists[i]) < max_len:
                list_of_lists[i].append([0,0,0,0,0,0])
    return list_of_lists

max_len = 30
process_data("merci", max_len, 17,5)
process_data("fini", max_len, 17,5)
process_data("maman", max_len, 17,5)
process_data("please", max_len, 17,5)
process_data("manger", max_len, 17,5)
process_data("livre", max_len, 17,5)

class BidirectionnalLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BidirectionnalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Linear(hidden_size*2, num_classes)
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).requires_grad_()
        out, _ = self.lstm(x, (h0, c0))
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
    
    def fit(self,data,epoch):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        for e in range(epoch):
            for i in range(len(data)):
                optimizer.zero_grad()
                output = self.forward(torch.tensor(data[i]).float())
                loss = criterion(output, torch.tensor([0,1]).long())
                loss.backward()
                optimizer.step()
            print(f"Epoch {e+1}/{epoch} : loss = {loss.item():.4f}")

    def predict(self,data):
        return self.forward(torch.tensor(data).float()).argmax().item()
    
    def test(self,data):
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(len(data)):
                output = self.forward(torch.tensor(data[i]).float())
                if output.argmax().item() == 0:
                    correct += 1
                total += 1
        print(f"Accuracy = {correct/total:.4f}")
    
# use the model to train on the data
merci_train_data = np.load("merci_train_data.npy")
fini_train_data = np.load("fini_train_data.npy")
maman_train_data = np.load("maman_train_data.npy")
please_train_data = np.load("please_train_data.npy")
manger_train_data = np.load("manger_train_data.npy")
livre_train_data = np.load("livre_train_data.npy")


merci_test_data = np.load("merci_test_data.npy")
fini_test_data = np.load("fini_test_data.npy")
maman_test_data = np.load("maman_test_data.npy")
please_test_data = np.load("please_test_data.npy")
manger_test_data = np.load("manger_test_data.npy")
livre_test_data = np.load("livre_test_data.npy")

train_data = np.concatenate((merci_train_data, fini_train_data, maman_train_data, please_train_data, manger_train_data, livre_train_data), axis=0)
train_targets = np.concatenate((np.zeros(len(merci_train_data)), 
                                np.ones(len(fini_train_data)), np.ones(len(maman_train_data))*2,
                                np.ones(len(please_train_data))*3, np.ones(len(manger_train_data))*4, np.ones(len(livre_train_data))*5), axis=0)
test_data = np.concatenate((merci_test_data, 
                            fini_test_data, maman_test_data, please_test_data, manger_test_data, livre_test_data), axis=0)
test_targets = np.concatenate((np.zeros(len(merci_test_data)), 
                              np.ones(len(fini_test_data)), np.ones(len(maman_test_data))*2, np.ones(len(please_test_data))*3, 
                              np.ones(len(manger_test_data))*4, np.ones(len(livre_test_data))*5), axis=0)
# split the data into train and test

x_train = torch.tensor(train_data, dtype=torch.float32)
y_train = torch.tensor(train_targets, dtype=torch.long)
x_test = torch.tensor(test_data, dtype=torch.float32)
y_test = torch.tensor(test_targets, dtype=torch.long)
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
BATCH_SIZE = 14
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=80, shuffle=True)
# train the model
model = BidirectionnalLSTM(6, 64, 3, 6)
# model.fit(x_train, 10)
criterion = torch.nn.CrossEntropyLoss()
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


loss_values = []
num_epochs = 500
y_pred_list = []
y_true_list = []
for epoch in range(num_epochs):
    # if epoch > 1 and epoch % 200 == 0:
    #     lr /= 2
    #     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data
        targets = targets
        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        loss_values.append(loss.item())
        # backward
        optimizer.zero_grad()
        loss.backward()
        # gradient descent or adam step
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs} : loss = {loss.item():.4f}")
    for batch_idx, (data, targets) in enumerate(test_loader):
        # compute the accuracy on the test set
        # confusion matrix
        scores = model(data)
        _, predictions = scores.max(1)
        num_correct = (predictions == targets).sum()
        running_test_acc = float(num_correct)/float(data.shape[0])
    print(f"Test accuracy = {running_test_acc:.4f}")
for batch_idx, (data, targets) in enumerate(test_loader):
    # compute the accuracy on the test set
    # confusion matrix
    scores = model(data)
    _, predictions = scores.max(1)
    y_pred_list.append(predictions.numpy())
    y_true_list.append(targets.numpy())
    num_correct = (predictions == targets).sum()
    running_test_acc = float(num_correct)/float(data.shape[0])
print(f"Test final accuracy = {running_test_acc:.4f}")


y_pred_list = [a.squeeze().tolist() for a in y_pred_list][0]
y_true_list = [a.squeeze().tolist() for a in y_true_list][0]

classes = ["merci", 
           "fini", "maman", "please", "manger", "livre"]
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pdb

cf_matrix = confusion_matrix(y_true_list, y_pred_list)
df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix, axis=1)[:None], index = [i for i in classes],
                    columns = [i for i in classes])
plt.figure(figsize = (10,7))
sns.heatmap(df_cm, annot=True)#, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r', xticklabels=classes, yticklabels=classes);
plt.savefig("confusion_matrix.png")


# save the model
torch.save(model.state_dict(), "model.pt")