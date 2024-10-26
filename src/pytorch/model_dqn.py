import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define the DQN model
class DQN(nn.Module):
    def __init__(self, input_dims, n_actions, fc1_dims):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        fc_input_dims = self.calculate_fc_input_dims(input_dims)
        self.fc = nn.Linear(fc_input_dims, fc1_dims)
        self.out = nn.Linear(fc1_dims, n_actions)

    def forward(self, state):
        conv1 = torch.relu(self.conv1(state))
        conv2 = torch.relu(self.conv2(conv1))
        conv3 = torch.relu(self.conv3(conv2))

        flatten = conv3.view(conv3.size()[0], -1)
        flat1 = torch.relu(self.fc(flatten))
        actions = self.out(flat1)

        return actions

    def calculate_fc_input_dims(self, input_dims):
        state = torch.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)

        return int(torch.prod(torch.tensor(dims.size())))


# Build the DQN model
def build_dqn(n_actions, input_dims, fc1_dims):
    model = DQN(input_dims, n_actions, fc1_dims).to(device)
    return model
