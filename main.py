import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def scale(data):
    max_value = data.max()
    min_value = data.min()
    data = (data - min_value)/(max_value-min_value)
    data = 2*data - 1
    return data, max_value, min_value


def inver_scale(data, max_value, min_value):
    data = (data+1)/2*(max_value-min_value) + min_value
    return data


def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


def readFile(filepath, look_back):
    data_csv = pd.read_csv(filepath, names=['open', 'high', 'low', 'close'])
    data_csv = data_csv['open']
    dataset = data_csv.values.astype('float32')

    data, max_value, min_value = scale(dataset)
    data_X, data_Y = create_dataset(data, look_back)
    return data_X, data_Y, max_value, min_value


def splitData(data_X, data_Y, look_back):
    train_size = int(len(data_X) * 0.7)
    test_size = len(data_X) - train_size
    print('train size : ', train_size)
    print('test size : ', test_size)
    train_X = data_X[:train_size]
    train_Y = data_Y[:train_size]
    test_X = data_X[train_size:]
    test_Y = data_Y[train_size:]

    train_X = train_X.reshape(-1, 1, look_back)
    train_Y = train_Y.reshape(-1, 1, 1)
    test_X = test_X.reshape(-1, 1, look_back)
    test_Y = test_Y.reshape(-1, 1, 1)

    train_x = torch.from_numpy(train_X)
    train_y = torch.from_numpy(train_Y)
    test_x = torch.from_numpy(test_X)
    test_y = torch.from_numpy(test_Y)
    return (train_x, train_y), (test_x, test_y)


class GRUNet(nn.Module):
    def __init__(self, input_size):
        super(GRUNet, self).__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=64,
            num_layers=3,
            batch_first=True,
            bidirectional=True
        )

        self.out = nn.Sequential(
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x, _ = self.rnn(x)
        s, b, h = x.shape
        x = x.view(s*b, h)
        x = self.out(x)
        x = x.view(s, b, -1)

        return x


def RMSELoss(yhat, y):
    return torch.sqrt(torch.mean((yhat-y)**2))


def train(net, criterion, optimizer, train_data, test_data, epochs):
    min_loss = 1.0
    train_loss = None
    test_loss = None

    train_x, train_y = train_data
    test_x, test_y = test_data
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=60)
    for e in range(epochs):
        net.train()
        var_x, var_y = Variable(train_x).to(
            device), Variable(train_y).to(device)
        out = net(var_x)
        optimizer.zero_grad()
        loss = criterion(out, var_y)
        loss.backward()
        optimizer.step()
        train_loss = loss.item()

        net.eval()
        with torch.no_grad():
            var_test_x, var_test_y = Variable(test_x).to(
                device), Variable(test_y).to(device)
            test_out = net(var_test_x)
            loss = criterion(test_out, var_test_y)
            test_loss = loss.item()
        scheduler.step(test_loss)
        if test_loss < min_loss:
            torch.save(net, './best_model.pt')
            print('Save model : loss :', test_loss)
            min_loss = test_loss
        if (e + 1) % 5 == 0:
            print('Epoch: {}, Train Loss: {:.5f}, Test Loss: {:.5f}'.format(
                e + 1, train_loss, test_loss))


def predict(training, testing, look_back, output):
    model = torch.load('./best_model.pt').to('cpu')
    model.eval()
    gt_csv = pd.read_csv(testing, names=['open', 'high', 'low', 'close'])
    gt_data = np.array(gt_csv['open'], dtype='float32')

    data_csv = pd.read_csv(
        training, names=['open', 'high', 'low', 'close'])
    dataset = np.array(data_csv['open'], dtype='float32')

    test_data = np.append(dataset[-look_back:], gt_data)
    test_data, max_value, min_value = scale(test_data)

    pred = []
    for i in range(20):
        input_data = test_data[i:i+look_back]
        input_data = input_data.reshape(-1, 1, look_back)
        input_data = torch.from_numpy(input_data)
        pred_test = model(input_data)

        pred.append(pred_test[-1].data.numpy().squeeze())

    pred = inver_scale(np.array(pred), max_value, min_value)

    action = []
    status = 0
    for i in range(len(pred)-3):
        future = pred[i+3] - pred[i+2]
        if i == len(pred)-4:
            action.append(0)
            action.append(0)
            action.append(0)            
        elif status == 0:
            if future > 0:
                action.append(1)
                status += 1
            elif future < 0:
                action.append(-1)
                status -= 1
            else:
                action.append(0)
        elif status == 1:
            if future < 0:
                action.append(-1)
                status -= 1
            else:
                action.append(0)
        elif status == -1:
            if future > 0:
                action.append(1)
                status += 1
            else:
                action.append(0)
    with open(output, 'w') as f:
        for a in action:
            f.write(str(a) + '\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                        default='training_data.csv',
                        help='input training data file name')
    parser.add_argument('--testing',
                        default='testing_data.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()

    look_back = 10
    file_name = './train.csv'
    net = GRUNet(look_back).to(device)
    criterion = RMSELoss
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    data_X, data_Y, max_value, min_value = readFile(args.training, look_back)

    train_data, test_data = splitData(data_X, data_Y, look_back)

    train(net, criterion, optimizer, train_data, test_data, 500)
    predict(args.training, args.testing, look_back, args.output)
