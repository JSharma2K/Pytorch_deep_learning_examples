# Description: This file contains the class Experiments which is used
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader


class Dataset:

    def __init__(self, path,experiment_str):
        '''
        path: path to the directory containing the data files
        experiment_str: string to identify the experiment folder
        '''
        self.path = path
        self.files = []
        self.data_frames = []
        self.experiment_str=experiment_str

    def get_files_in_folder(self):
        """
        Get all files present in the given directory.
        """
        if os.path.isdir(self.path):
            for root, dirs, files in os.walk(self.path):
                for file in files:
                    # Ignore files starting with '~' or named '.DS_Store'
                    if not file.startswith("~") and file != '.DS_Store':
                        self.files.append(os.path.join(root, file))
        else:
            raise ValueError(f"Provided path: {self.path} is not a directory")

    def load_dataset(self):
        """
        Load data from all .csv and .xlsx files present in the directory.
        """
        self.get_files_in_folder()
        path_to_load=[file for file in self.files if self.experiment_str in file]
        for file_ in path_to_load:
            if os.path.isdir(file_):
                for root, dirs, files in os.walk(path_to_load):
                    for file in files:
                        if not file.startswith("~") and file != '.DS_Store':
                            self.files.append(os.path.join(root, file))
                            self.data_frames.append(pd.read_csv(file) if file.endswith(".csv") else pd.read_excel(file))
            else:
                self.data_frames.append(pd.read_csv(file_) if file_.endswith(".csv") else pd.read_excel(file_))
        return self.data_frames


# To use the Dataset class:

# directory = 'path_to_your_directory'
# dataset = Dataset(directory)
# dataset.load_data()

# Now, dataset.data_frames will contain a list of all the pandas DataFrames
# loaded from the .csv and .xlsx files present in the given directory.


class Experiment:
    def __init__(self,dataset_path,experiment_str):
        self.experiment_str = experiment_str
        self.dataset_path=dataset_path
        self.dataset=Dataset(self.dataset_path,self.experiment_str)

    def run_experiment(self):
        print("Running experiment: ", self.experiment_str)
        if "linear_regression_example" in self.experiment_str:
            self.pytorch_linear_regression_example(data_exploration=False)

    def pytorch_linear_regression_example(self,data_exploration=False):
        Dataframes=self.dataset.load_dataset()
        if len(Dataframes)==1:
            linear_regression_dataset=Dataframes[0]
        else:
            linear_regression_dataset=pd.concat(Dataframes)

        if data_exploration:
            # doing some data exploration
            print(linear_regression_dataset.info())
            print(linear_regression_dataset.describe())
            sns.pairplot(linear_regression_dataset)
            print(linear_regression_dataset.corr())
            print(linear_regression_dataset.head())
            plt.hist(linear_regression_dataset['Price'], bins=30)
            plt.show()
            print("Max price is: {} ".format(linear_regression_dataset['Price'].max()))
            print("Min price is: {} ".format(linear_regression_dataset['Price'].min()))
            print("Median price is: {} ".format(linear_regression_dataset['Price'].median()))
            print("difference between max and min price is: {} ".format(linear_regression_dataset['Price'].max()-linear_regression_dataset['Price'].min()))


        # Split the data into training and test sets using the train_test_split function from scikit-learn.
        # We will use 80% of the data for training and 20% for testing.
        # We will also use the random_state parameter to ensure that we get the same split every time we run the code.

        features = linear_regression_dataset.loc[:, ~linear_regression_dataset.columns.isin(['Price','Address'])]
        target = linear_regression_dataset['Price']

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        feature_scaler = MinMaxScaler()
        X_train = feature_scaler.fit_transform(X_train)
        X_test = feature_scaler.transform(X_test)

        target_scaler = MinMaxScaler()
        y_train=target_scaler.fit_transform(y_train.values.reshape(-1, 1))
        y_test=target_scaler.transform(y_test.values.reshape(-1, 1))
        #convert the data into tensors using the torch.from_numpy function which will allow us to use the data in PyTorch using data loaders.

        features_tensor_train = torch.from_numpy(X_train).float()
        targets_tensor_train = torch.from_numpy(y_train).float()
        features_tensor_test = torch.from_numpy(X_test).float()
        targets_tensor_test = torch.from_numpy(y_test).float()

        #create a dataset class to be able to use the data loaders
        class dataset(Dataset):
            def __init__(self, x, y):
                self.x = x
                self.y = y

            def __getitem__(self, idx):
                return self.x[idx], self.y[idx]

            def __len__(self):
                return self.x.shape[0]

        #create train and test datasets using the dataset class and a dataloader with batch size 64

        train_set = dataset(features_tensor_train, targets_tensor_train)
        test_set = dataset(features_tensor_test, targets_tensor_test)

        train_loader = DataLoader(train_set, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
        class LinearRegressionModel(torch.nn.Module):
            def __init__(self):
                super(LinearRegressionModel, self).__init__()
                self.linear = torch.nn.Linear(5, 1)

            def forward(self, x):
                y_pred = self.linear(x)
                return y_pred

        learning_rate = 0.01
        epochs = 100
        MLP = LinearRegressionModel()
        optimizer = torch.optim.SGD(MLP.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss(reduction='mean')

        #train the model
        loss_per_epoch = []
        for epoch in range(epochs):
            for x_train, y_train in train_loader:
                optimizer.zero_grad()
                y_pred = MLP(x_train)
                loss = criterion(y_pred, y_train)
                loss.backward()
                optimizer.step()
            loss_per_epoch.append(loss.item())
            print('epoch {}, loss {}'.format(epoch, loss.item()))

        plt.plot(loss_per_epoch)
        plt.show()

        #test the model
        with torch.no_grad():
            y_pred_vals=[]
            for x_test, y_test in test_loader:
                y_pred = MLP(x_test)
                loss = criterion(y_pred, y_test)
                y_pred_vals.append(y_pred.squeeze())
                print('loss on test set: {}'.format(loss.item()))

            y_pred_vals=torch.cat(y_pred_vals)
            y_test_vals=torch.cat([y_test for x_test, y_test in test_loader]).squeeze()

            #plot the results
            plt.scatter(y_test_vals, y_pred_vals)
            plt.xlabel('Actual Price')
            plt.ylabel('Predicted Price')
            plt.title('Actual vs Predicted Price')
            plt.show()

            #calculate the r2 score

            def r_squared(y_true, y_pred):
                ssr = torch.sum((y_true - y_pred) ** 2)
                sst = torch.sum((y_true - torch.mean(y_true)) ** 2)
                r2_score = 1 - ssr / sst
                return r2_score
            r2 = r_squared(y_test_vals, y_pred_vals)
            print('r2 score: {}'.format(r2))














