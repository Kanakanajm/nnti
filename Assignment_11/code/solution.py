import random
import torch
import pickle
import os
from torch import nn
from torch.utils.data import Dataset, DataLoader
from corpus import load, IMDBTrainCorpus, IMDBTestCorpus, IMDBTrainLabels, IMDBTestLabels


w2vmodel = load()

class IMDBDataset(Dataset):
    def __init__(self, corpus, labels):
        self.labels = [l for l in labels]
        self.documents = [doc for doc in corpus]

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        doc = self.documents[idx]
        label = self.labels[idx]
        # has shape: (L, H_in) where L is the number of words in the document
        # and H_in is the dimension of each vectorised word
        doc_vec = torch.stack([torch.tensor(w2vmodel.wv[w]) for w in doc if w in w2vmodel.wv.key_to_index])
        return doc_vec, label

BATCH_SIZE = 64

class IMDBSentimentModel(nn.Module):
    def __init__(self):
        super(IMDBSentimentModel, self).__init__()
        self.rnn = nn.RNN(100, 200, batch_first=True)
        self.fc = nn.Linear(200, 10)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(1, batch_size, 200)
        return hidden
    
    def forward(self, x):
            batch_size = x.size(0)
            # hidden = self.init_hidden(batch_size)
            hidden = torch.randn(1, 200)
            out, _ = self.rnn(x, hidden)
            
            return self.fc(out[-1, :])

def save_pickle(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)

def load_pickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# def load_else_dump(filename, obj):
#     if os.path.isfile(filename):
#         return load_pickle(filename)
#     else:
#         save_pickle(obj, filename)
#         return obj

TRAIN_DATA_PICKLE = 'train_data.pkl'
TEST_DATA_PICKLE = 'test_data.pkl'




train_data = load_pickle(TRAIN_DATA_PICKLE)
# test_data = load_pickle(TEST_DATA_PICKLE)
print("Dataset loaded")

# train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
print("Dataloader loaded")


# Define hyperparameters
# n_epochs = 2
lr=0.01

model = IMDBSentimentModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

print("Start training")
# for epoch in range(n_epochs):
#     for i, data in enumerate(train_dataloader):
#         optimizer.zero_grad() # Clears existing gradients from previous epoch

#         inputs, labels = data
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward() 
#         optimizer.step()
    
#         if i%10 == 0:
#             print('Epoch: {}/{} Training Sample {}.............'.format(epoch, n_epochs, i), end=' ')
#             print("Loss: {:.4f}".format(loss.item()))

for i in range(1000):
    idx = random.randint(0, 25000)
    optimizer.zero_grad() # Clears existing gradients from previous epoch
    input, label = train_data[idx]
    output = model(input)
    loss = criterion(output, torch.eye(10)[label - 1])
    loss.backward() 
    optimizer.step()

    if i%10 == 0:
        print('Training Sample {}.............'.format(i), end=' ')
        print("Loss: {:.4f}".format(loss.item()))