import torch
import pickle

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F

SAMPLE_RATE = 100
IN_FEATURES = 6

# Input feature would be a tensor with 6 features with the time of 1 second
class GestureLSTMModule(torch.nn.Module):
    def __init__(self):
        super(GestureLSTMModule, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=IN_FEATURES, hidden_size=32, num_layers=2, batch_first=True)
        self.fc = torch.nn.Linear(32, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        x = F.sigmoid(x)
        return x
    

class GestureDNNModule(torch.nn.Module):
    def __init__(self):
        super(GestureDNNModule, self).__init__()
        self.fc1 = torch.nn.Linear(IN_FEATURES * SAMPLE_RATE, 32)
        self.fc2 = torch.nn.Linear(32, 1)
        
    def forward(self, x):
        x = x.view(-1, IN_FEATURES * SAMPLE_RATE)
        x = F.gelu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x
    
class GestureDataset(Dataset):
    def __init__(self, data, device='cpu'):
        self.data = data
        self.device = device
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.Tensor(self.data[idx][0]).float().to(self.device), torch.Tensor([self.data[idx][1]]).float().to(self.device)
    
    
def pre_process():
    pass
    
if __name__ == '__main__':
    positive, negative = pickle.load(open('data/positive.pkl', 'rb')), pickle.load(open('data/negative.pkl', 'rb'))
    dataset = [(x, 1) for x in positive] + [(x, 0) for x in negative]
    training, testing = train_test_split(dataset, test_size=0.2)
    train_dataset, test_dataset = GestureDataset(training), GestureDataset(testing)
    train_dataloader, test_dataloader = DataLoader(train_dataset, batch_size=32), DataLoader(test_dataset, batch_size=32)
    
    model = GestureLSTMModule()
    for p in model.parameters():
        _ = p.requires_grad_(True)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    accumulated_loss = 0
    training_steps = 0
    num_epochs = 10
    
    for epoch in range(num_epochs):
        for x, y in train_dataloader:
            outputs = model(x)
            loss = torch.nn.BCELoss(outputs, y)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            accumulated_loss += loss.item()
            training_steps += 1
            if training_steps % 100 == 0:
                print(f'Loss: {accumulated_loss / training_steps}')
                predictions, labels = [], []
                eval_loss = 0
                for x, y in test_dataloader:
                    outputs = model(x)
                    predictions.extend(outputs)
                    labels.extend(y)
                    eval_loss += torch.nn.BCELoss(outputs, y).item()
                predictions = torch.cat(predictions, dim=0)
                print(f'Eval loss: {eval_loss / len(test_dataloader)}')
                print(f'Accuracy: {torch.sum((predictions > 0.5) == labels) / len(labels)}')
                
    torch.save(model.state_dict(), 'model.pt')