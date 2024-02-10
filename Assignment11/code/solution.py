import torch
import pickle
# import os
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
# from corpus import load, IMDBTrainCorpus, IMDBTestCorpus, IMDBTrainLabels, IMDBTestLabels
from corpus import load

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

def pad_collate(batch):
  (xx, yy) = zip(*batch)
  xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)

  return xx_pad, torch.Tensor(yy)

class IMDBSentimentModel(nn.Module):
    def __init__(self):
        super(IMDBSentimentModel, self).__init__()
        self.rnn = nn.RNN(100, 10, batch_first=True)
        # self.fc = nn.Linear(1, 10)

    def init_hidden(self, batch_size):
        hidden = torch.randn(1, batch_size, 10)
        return hidden
    
    def forward(self, x):
            batch_size = x.size(0)
            hidden = self.init_hidden(batch_size)

            out, _ = self.rnn(x, hidden)
            
            return out[:, -1, :]

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
test_data = load_pickle(TEST_DATA_PICKLE)
print("Dataset loaded")

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)

MODEL_FILE = 'imdb_sentiment.model'

def train():
    # Define hyperparameters
    n_epochs = 2
    lr=0.001

    model = IMDBSentimentModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    model.train()
    print("Start Training")
    for epoch in range(n_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):
            optimizer.zero_grad()

            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, nn.functional.one_hot(labels.long() - 1, num_classes = 10).float())
            loss.backward() 
            optimizer.step()

            running_loss += loss.item()
            if i%100 == 99:
                print('Epoch: {}/{} Training Sample {}.............'.format(epoch+1, n_epochs, i+1), end=' ')
                print("Loss: {:.4f}".format(running_loss / 100))
                running_loss = 0.0

    print('Finished Training')
    torch.save(model, MODEL_FILE)
    print('Saved model to' + MODEL_FILE)
    return model

def evaluate(model: IMDBSentimentModel):
    print('Evaluating')
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(train_dataloader):
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    print('Accuracy: %d %%' % (
        100 * correct / total))
    
model = train()
# model = torch.load(MODEL_FILE)
evaluate(model)




