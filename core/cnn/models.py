import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNMatchModel(nn.Module):
    def __init__(self, input_matrix_size1, input_matrix_size2, mat1_channel1, mat1_kernel_size1,
                 mat1_channel2, mat1_kernel_size2, mat1_hidden, mat2_channel1, mat2_kernel_size1,
                 mat2_hidden):
        super(CNNMatchModel, self).__init__()
        self.mat_size1 = input_matrix_size1
        self.mat_size2 = input_matrix_size2

        self.conv1_1 = nn.Conv2d(1, mat1_channel1, mat1_kernel_size1)  # n*mat1_channel1*(input_matrix_size1-mat1_kernel_size1+1)*(input_matrix_size1-mat1_kernel_size1+1)
        self.conv1_2 = nn.Conv2d(mat1_channel1, mat1_channel2, mat1_kernel_size2)  # n*mat1_channel2*(input_matrix_size1-mat1_kernel_size1-mat1_kernel_size2+2)*(input_matrix_size1-mat1_kernel_size1-mat1_kernel_size2+2)
        self.mat1_flatten_dim = mat1_channel2*((input_matrix_size1-mat1_kernel_size1-mat1_kernel_size2+2)**2)
        self.fc1_1 = nn.Linear(self.mat1_flatten_dim, mat1_hidden)

        self.conv2_1 = nn.Conv2d(1, mat2_channel1, mat2_kernel_size1)  # n*mat2_channel1*(input_matrix_size2-mat2_kernel_size1+1)*(input_matrix_size2-mat2_kernel_size1+1)
        self.mat2_flatten_dim = mat2_channel1*((input_matrix_size2-mat2_kernel_size1+1)**2)
        self.fc2_1 = nn.Linear(self.mat2_flatten_dim, mat2_hidden)

        self.fc_out = nn.Linear(mat1_hidden+mat2_hidden, 2)

    def forward(self, batch_matrix1, batch_matrix2):
        batch_matrix1 = batch_matrix1.unsqueeze(1)
        batch_matrix2 = batch_matrix2.unsqueeze(1)

        mat1 = F.relu(self.conv1_1(batch_matrix1))
        mat1 = F.relu(self.conv1_2(mat1))
        mat1 = mat1.view(-1, self.mat1_flatten_dim)
        hidden1 = self.fc1_1(mat1)

        mat2 = F.relu(self.conv2_1(batch_matrix2))
        mat2 = mat2.view(-1, self.mat2_flatten_dim)
        hidden2 = self.fc2_1(mat2)

        hidden = torch.cat((hidden1, hidden2), 1)
        out = self.fc_out(hidden)
        return F.log_softmax(out, dim=1), F.softmax(out, dim=1)
