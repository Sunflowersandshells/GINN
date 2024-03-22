#The code was developed by Wanli Xie. For any inquiries,
# please feel free to contact us at wanlix2021@163.com.
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, \
    median_absolute_error, mean_squared_log_error
parser = argparse.ArgumentParser('GINN demo')
parser.add_argument('--learning_rate', type=int, default=0.001)
parser.add_argument('--test_number', type=int, default=4)
parser.add_argument('--n_steps', type=int, default=2)
parser.add_argument('--hidden_dim', type=int, default=30)
parser.add_argument('--epoch_number', type=int, default=2000)
parser.add_argument('--data_set', type=str, default='Example_data.xlsx')
parser.add_argument('--training_set_error', type=str, default='train_data_index_all.csv')
parser.add_argument('--training_set_values', type=str, default='y_train_predict_all.csv')
parser.add_argument('--test_set_error', type=str, default='y_test_data_index_all.csv')
parser.add_argument('--test_set_values', type=str, default='y_test_predict_all.csv')
parser.add_argument('--viz', action='store_true')
args = parser.parse_args()

Case_name = 'Case'
train_data_index_all = []
y_train_predict_all = []
test_data_index_all = []
y_test_predict_all = []

class CustomLossWithParams(torch.nn.Module):
    def __init__(self, param1, param2, param3):
        super(CustomLossWithParams, self).__init__()
        self.param1 = param1  # Parameters of grey model
        self.param2 = param2  # Parameters of grey model
        self.param3 = param3  # Weighted coefficient
    def forward(self, y_pred, y_true):
        loss_NN = torch.mean((y_pred[1:] - y_true[1:]) ** 2)  # Calculate MSEloss from the second number
        x1 = torch.cumsum(y_pred, dim=0, dtype=torch.float32)
        x_neighbor_mean = (x1[:-1] + x1[1:]) / 2.0
        loss_GM = torch.mean(torch.abs(y_pred[1:] + self.param1 * x_neighbor_mean - self.param2))
        loss_all = (1 - self.param3) * loss_NN + self.param3 * loss_GM
        return loss_all

class GModel:
    def __init__(self, original_data, prediction_steps):
        """
        Initialize the GModel.
        :param original_data: The original data.
        :param prediction_steps: Number of steps to predict forward.
        """
        self.original_data = original_data
        self.prediction_steps = prediction_steps


    def calculate_results(self):
        n = len(self.original_data)
        cumulative_data = np.cumsum(self.original_data)
        z = np.zeros(n - 1)
        for i in range(n - 1):
            z[i] = 0.5 * (cumulative_data[i] + cumulative_data[i + 1])
        B = [-z, [1] * (n - 1)]
        Y = self.original_data[1:]
        u = np.dot(np.linalg.inv(np.dot(B, np.transpose(B))), np.dot(B, Y))
        x1_solve = np.zeros(n)
        x0_solve = np.zeros(n)
        x1_solve[0] = x0_solve[0] = self.original_data[0]
        for i in range(1, n):
            x1_solve[i] = (self.original_data[0] - u[1] / u[0]) * np.exp(-u[0] * i) + u[1] / u[0]
        for i in range(1, n):
            x0_solve[i] = x1_solve[i] - x1_solve[i - 1]
        return np.array(x0_solve), x1_solve, u

def Mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100
# SMAPE

def Smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100
# Define the model

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


#Splitting data method, referenced from Brownlee's work
#Brownlee J. Deep learning for time series forecasting: predict the future with MLPs, CNNs and LSTMs in Python[M]. Machine Learning Mastery, 2018.
def split_sequence(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return torch.tensor(X), torch.tensor(y)


class Visualizer:
    def __init__(self, y_train, y_train_predict):
        self.y_train = y_train.numpy() if isinstance(y_train, torch.Tensor) else y_train
        self.y_train_predict = y_train_predict.numpy() if isinstance(y_train_predict, torch.Tensor) else y_train_predict

    def plot_comparison(self):
        if args.viz:


            plt.figure(figsize=(10, 6))
            plt.plot(self.y_train, label="Actual Values", color='b')
            plt.plot(self.y_train_predict, label="Predicted Values", color='r', linestyle='--')
            plt.xlabel('Sample Index')
            plt.ylabel('Value')
            plt.title('Comparison of Actual vs Predicted Values')
            plt.legend()
            plt.show()

    def scatter_plot(self):
        if args.viz:

            plt.figure(figsize=(10, 6))
            plt.scatter(range(len(self.y_train)), self.y_train, label='Actual Values', color='b')
            plt.scatter(range(len(self.y_train_predict)), self.y_train_predict, label='Predicted Values', color='r')
            plt.xlabel('Sample Index')
            plt.ylabel('Value')
            plt.title('Scatter Plot of Actual vs Predicted Values')
            plt.legend()
            plt.show()

if __name__ == '__main__':
    all_data = pd.read_excel(args.data_set, header=None)
    all_data = all_data.astype('float64')
    all_data = all_data.dropna(axis=1)
    test_number = args.test_number
    n_steps = args.n_steps
    hidden_dim = args.hidden_dim
    model = Net(n_steps, hidden_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)


    for i_data in range(all_data.shape[0]):
        raw_seq = all_data.loc[i_data, :].values.tolist()
        train_set = raw_seq[:-test_number]
        test_set = raw_seq[-test_number:]
        X_data, y_data = split_sequence(raw_seq, n_steps)
        X_train = X_data[:-test_number, :]
        X_test = X_data[-test_number:, :]
        y_train = y_data[:-test_number]
        y_test = y_data[-test_number:]
        train_set_GM = train_set[n_steps:]
        gm = GModel(original_data=np.array(train_set_GM), prediction_steps=0)
        x0_solve, x1_solve, u = gm.calculate_results()
        test_predict_one_case = []


        for epoch in range(args.epoch_number):
            optimizer.zero_grad()
            y_train_predict = model(X_train.float())
            custom_loss = CustomLossWithParams(u[0], u[1], 0.1)
            loss = custom_loss(y_train_predict, y_train.float().view(-1, 1))
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            model.eval()
            y_train_predict = model(X_train.float())
            y_test_predict = model(X_test.float())


        print(Case_name + str(i_data + 1) + "      --------Train_error_start----------")
        y_train_predict = y_train_predict.flatten()
        y_test_predict = y_test_predict.flatten()

        visualizer = Visualizer(y_train, y_train_predict)
        visualizer.plot_comparison()
        visualizer.scatter_plot()

        mse_MLP = mean_squared_error(y_train, y_train_predict)
        rmse_MLP = mse_MLP ** 0.5
        mae_DTR = mean_absolute_error(y_train, y_train_predict)
        R2_MLP = r2_score(y_train, y_train_predict)
        mape_MLP = torch.mean(torch.abs(y_train_predict - y_train) / y_train) * 100
        evs_MLP = explained_variance_score(y_train, y_train_predict)
        msle_MLP = mean_squared_log_error(y_train, y_train_predict)
        train_data_index = pd.DataFrame([mape_MLP, mse_MLP, mae_DTR, rmse_MLP, R2_MLP, evs_MLP, msle_MLP],
                                        index=["mape", "mse", "mae", "rmse", "r2",
                                               "evs", "msle"])

        y_train_predict = pd.DataFrame(y_train_predict)
        train_data_index_all.append(train_data_index.values.tolist())
        y_train_predict_all.append(y_train_predict.values.tolist())
        print("Test error list as mse,rmse,mae,r2, mape,evs,msle")
        print([mse_MLP, rmse_MLP, mae_DTR, R2_MLP, mape_MLP, evs_MLP, msle_MLP])
        print(Case_name + str(i_data + 1) + "        --------Test_error_start----------")
        mse_MLP = mean_squared_error(y_test, y_test_predict)
        rmse_MLP = mse_MLP ** 0.5
        mae_DTR = mean_absolute_error(y_test, y_test_predict)
        R2_MLP = r2_score(y_test, y_test_predict)
        mape_MLP = torch.mean(torch.abs(y_test_predict - y_test) / y_test) * 100
        evs_MLP = explained_variance_score(y_test, y_test_predict)
        msle_MLP = mean_squared_log_error(y_test, y_test_predict)
        test_data_index = pd.DataFrame(
            [mape_MLP, mse_MLP, mae_DTR, rmse_MLP, R2_MLP, evs_MLP, msle_MLP],
            index=["mape", "mse", "mae", "rmse", "r2",
                   "evs", "msle"])


        y_test_predict = pd.DataFrame(y_test_predict)
        test_data_index_all.append(test_data_index.values.tolist())
        y_test_predict_all.append(y_test_predict.values.tolist())
        print("Test error list as mse,rmse,mae,r2, mape,evs,msle")
        print([mse_MLP, rmse_MLP, mae_DTR, R2_MLP, mape_MLP, evs_MLP, msle_MLP])
    train_data_index_all_pd = pd.DataFrame(np.concatenate(train_data_index_all, axis=1),
                                           index=["mape", "mse", "mae", "rmse", "r2",
                                                  "evs", "msle"])

    train_data_index_all_pd.to_csv(args.training_set_error)
    y_train_predict_all_pd = pd.DataFrame(np.concatenate(y_train_predict_all, axis=1))
    y_train_predict_all_pd.to_csv(args.training_set_values)
    y_test_predict_all_pd = pd.DataFrame(np.concatenate(y_test_predict_all, axis=1))
    y_test_predict_all_pd.to_csv(args.test_set_values)
    test_data_index_all_pd = pd.DataFrame(np.concatenate(test_data_index_all, axis=1),
                                          index=["mape", "mse", "mae", "rmse", "r2",
                                                 "evs", "msle"])
    test_data_index_all_pd.to_csv(args.test_set_error)

