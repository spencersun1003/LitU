import torch
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F

SAMPLE_RATE = 10
IN_FEATURES = 6
TIME_WINDOW = 2

normalize = {
    'mean': {
        'ax': -0.04649547031947544,
        'ay': -0.7898887034824916,
        'az': -0.1780448477608817,
        'gx': -0.36303026172300984,
        'gy': -0.03285850599781901,
        'gz': -0.05809160305343511
    },
    'std': {
        'ax': 0.2560270511848388,
        'ay': 0.4629416712847358,
        'az': 0.3843875035875543,
        'gx': 6.684420876973939,
        'gy': 19.601419629629802,
        'gz': 6.819778407387476,
    }
}

# Input feature would be a tensor with 6 features with the time of 1 second
class GestureLSTMModule(torch.nn.Module):
    def __init__(self):
        super(GestureLSTMModule, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=IN_FEATURES, hidden_size=32, num_layers=2, batch_first=True)
        self.fc = torch.nn.Linear(32, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        x = F.sigmoid(x)
        return x
    

class GestureDNNModule(torch.nn.Module):
    def __init__(self):
        super(GestureDNNModule, self).__init__()
        self.fc1 = torch.nn.Linear(IN_FEATURES * SAMPLE_RATE * TIME_WINDOW, 32)
        self.fc2 = torch.nn.Linear(32, 1)
        
    def forward(self, x):
        x = x.view(-1, IN_FEATURES * SAMPLE_RATE * TIME_WINDOW)
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
    
    
def pre_process(data):
    data = data[200:-200]
    ax, ay, az, gx, gy, gz = np.array([d[0][0] for d in data]), np.array([d[0][1] for d in data]), np.array([d[0][2] for d in data]), np.array([d[1][0] for d in data]), np.array([d[1][1] for d in data]), np.array([d[1][2] for d in data])
    for d, name in zip([ax, ay, az, gx, gy, gz], ['ax', 'ay', 'az', 'gx', 'gy', 'gz']):
        d -= normalize['mean'][name]
        d /= normalize['std'][name]
    returned_data = []
    for i in range(0, len(data) - SAMPLE_RATE * TIME_WINDOW, SAMPLE_RATE):
        returned_data.append(np.array([ax[i:i + SAMPLE_RATE * TIME_WINDOW], ay[i:i + SAMPLE_RATE * TIME_WINDOW], az[i:i + SAMPLE_RATE * TIME_WINDOW], gx[i:i + SAMPLE_RATE * TIME_WINDOW], gy[i:i + SAMPLE_RATE * TIME_WINDOW], gz[i:i + SAMPLE_RATE * TIME_WINDOW]]).T)
    return returned_data
    

if __name__ == '__main__':
    positive, negative = pickle.load(open('data/positive.pkl', 'rb')), pickle.load(open('data/negative.pkl', 'rb'))
    positive, negative = pre_process(positive), pre_process(negative)
    dataset = [(x, 1) for x in positive] + [(x, 0) for x in negative]
    training, testing = train_test_split(dataset, test_size=0.2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset, test_dataset = GestureDataset(training, device), GestureDataset(testing, device)
    train_dataloader, test_dataloader = DataLoader(train_dataset, batch_size=32), DataLoader(test_dataset, batch_size=32)
    
    model = GestureDNNModule()
    for p in model.parameters():
        _ = p.requires_grad_(True)
    model.train()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()
    accumulated_loss = 0
    training_steps = 0
    num_epochs = 100
    
    best_f1 = 0
    for epoch in range(num_epochs):
        for x, y in train_dataloader:
            outputs = model(x)
            loss = criterion(outputs, y)
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
                    eval_loss += criterion(outputs, y).item()
                predictions = torch.cat(predictions, dim=0)
                predictions, labels = (predictions > 0.5).cpu(), torch.Tensor(labels).int()
                print(f'Eval loss: {eval_loss / len(test_dataloader)}')
                print(f'Accuracy: {torch.sum(predictions == labels) / len(labels)}')
                # Calculating precision, recall, f1
                precision = torch.sum(predictions * labels) / torch.sum(predictions)
                recall = torch.sum(predictions * labels) / torch.sum(labels)
                f1 = 2 * precision * recall / (precision + recall)
                print(f'Precision: {precision}, Recall: {recall}, F1: {f1}')
                if f1 > best_f1:
                    best_f1 = f1
                    torch.save(model.state_dict(), 'model.pt')