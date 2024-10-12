import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset  # Pytorch에서 데이터를 불러오고, 전처리하는 클래스
import numpy as np


class TestDataset(Dataset):
    def __init__(self, seq_data):
        self.X = []
        self.y = []
        for dic in seq_data:
            self.X.append(dic['value'])
            self.y.append(dic['key'])

    def __getitem__(self, index):
        data = self.X[index]
        label = self.y[index]
        return torch.Tensor(np.array(data)), torch.tensor(np.array(int(label)))

    def __len__(self):
            return len(self.X)


class AnomalyDetectionDataset(Dataset):
    def __init__(self, seq_data):
        self.dataset = []
        for data in seq_data:
            self.dataset.append(data)

    def __getitem__(self, index):
        data = self.dataset[index]
        return torch.Tensor(np.array(data))

    def __len__(self):
        return len(self.dataset)


class Anomaly_Calculator:
    def __init__(self, mean: np.array, std: np.array):
        assert mean.shape[0] == std.shape[0] and mean.shape[0] == std.shape[1], '평균과 분산의 차원이 똑같아야 합니다.'
        self.mean = mean
        self.std = std

    def __call__(self, recons_error: np.array):
        x = (recons_error - self.mean)
        return np.matmul(np.matmul(x, self.std), x.T)


class ClassificationModel(nn.Module):
    def __init__(self):
        super(ClassificationModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=100, hidden_size=172, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=172, hidden_size=256, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=256, hidden_size=512, num_layers=1, batch_first=True)
        self.lstm4 = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True)
        self.lstm5 = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        self.lstm6 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
        self.lstm7 = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, batch_first=True)
        self.lstm8 = nn.LSTM(input_size=32, hidden_size=16, num_layers=1, batch_first=True)
        self.fc = nn.Linear(16, 7)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = self.dropout1(x)
        x, _ = self.lstm4(x)
        x, _ = self.lstm5(x)
        x, _ = self.lstm6(x)
        x = self.dropout2(x)
        x, _ = self.lstm7(x)
        x, _ = self.lstm8(x)
        x = self.fc(x[:, -1, :])
        return x


class Swing016Model(nn.Module):
    def __init__(self):
        super(Swing016Model, self).__init__()
        self.lstm1 = nn.LSTM(input_size=100, hidden_size=2048, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=2048, hidden_size=512, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True)
        self.fc = nn.Linear(256, 10)
        self.dropout1 = nn.Dropout(0.3)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.dropout1(x)
        x, _ = self.lstm3(x)
        x = self.fc(x[:, -1, :])
        return x


class Swing2Model(nn.Module):
    def __init__(self):
        super(Swing2Model, self).__init__()
        self.lstm1 = nn.LSTM(input_size=100, hidden_size=256, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=64, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=64, hidden_size=16, num_layers=1, batch_first=True)
        self.fc = nn.Linear(16, 10)
        self.dropout1 = nn.Dropout(0.3)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.dropout1(x)
        x, _ = self.lstm3(x)
        x = self.fc(x[:, -1, :])
        return x


class Swing34Model(nn.Module):
    def __init__(self):
        super(Swing34Model, self).__init__()
        self.lstm1 = nn.LSTM(input_size=100, hidden_size=64, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=32, hidden_size=16, num_layers=1, batch_first=True)
        self.fc = nn.Linear(16, 10)
        self.dropout1 = nn.Dropout(0.3)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.dropout1(x)
        x, _ = self.lstm3(x)
        x = self.fc(x[:, -1, :])
        return x


class Swing5Model(nn.Module):
    def __init__(self):
        super(Swing5Model, self).__init__()
        self.lstm1 = nn.LSTM(input_size=100, hidden_size=256, num_layers=1, batch_first=True)
        self.lstm6 = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, 10)
        self.dropout1 = nn.Dropout(0.3)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm6(x)
        x = self.fc(x[:, -1, :])
        return x


class Encoder(nn.Module):

    def __init__(self, input_size=100, hidden_size=50, num_layers=2):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=0.3, bidirectional=False)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size)

        return hidden, cell


class Decoder(nn.Module):

    def __init__(self, input_size=100, hidden_size=50, output_size=100, num_layers=2):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=0.3, bidirectional=False)

        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        output, (hidden, cell) = self.lstm(x, hidden)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        prediction = self.fc(output)

        return prediction, (hidden, cell)


# LSTM Auto Encoder
class LSTMAutoEncoder(nn.Module):

    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 sequence_length: int = 1,
                 **kwargs) -> None:
        """
        :param input_dim: 변수 Tag 갯수
        :param latent_dim: 최종 압축할 차원 크기
        :param sequence length: sequence 길이
        :param kwargs:
        """

        super(LSTMAutoEncoder, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.sequence_length = sequence_length

        if "num_layers" in kwargs:
            num_layers = kwargs.pop("num_layers")
        else:
            num_layers = 1

        self.encoder = Encoder(
            input_size=input_dim,
            hidden_size=latent_dim,
            num_layers=num_layers,
        )
        self.reconstruct_decoder = Decoder(
            input_size=input_dim,
            output_size=input_dim,
            hidden_size=latent_dim,
            num_layers=num_layers,
        )

    def forward(self, src: torch.Tensor, **kwargs):
        batch_size, sequence_length, var_length = src.size()

        ## Encoder 넣기t
        encoder_hidden = self.encoder(src)

        inv_idx = torch.arange(sequence_length - 1, -1, -1).long()
        reconstruct_output = []
        temp_input = torch.zeros((batch_size, 1, var_length), dtype=torch.float).to(src.device)
        hidden = encoder_hidden
        for t in range(sequence_length):
            temp_input, hidden = self.reconstruct_decoder(temp_input, hidden)
            reconstruct_output.append(temp_input)
        reconstruct_output = torch.cat(reconstruct_output, dim=1)[:, inv_idx, :]

        return [reconstruct_output, src]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        input = args[1]

        ## MSE loss(Mean squared Error)
        loss = F.mse_loss(recons, input)
        return loss