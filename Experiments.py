# Description: This file contains the class Experiments which is used
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import random
import umap
import torch.optim as optim
import datetime
from torch.utils.data import WeightedRandomSampler
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score,silhouette_score, silhouette_samples
from sklearn.metrics import normalized_mutual_info_score
from joblib import Parallel, delayed
from torch.utils.data import Dataset, DataLoader
from utils import r_squared
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from torch.utils.tensorboard import SummaryWriter
import fasttext
from nltk.tokenize import RegexpTokenizer
from sklearn.cluster import KMeans
from torchmetrics.classification import MulticlassF1Score,MulticlassCohenKappa,MulticlassPrecision,MulticlassRecall

'''
def threeSum(nums):
    triplets = []
    for i in range(len(nums) - 1):
        for j in range(len(nums) - 1):
            if i == j:
                j += 1
            duo_sum = nums[i] + nums[j]
            sum_diff = 0 - duo_sum
            compare_list = [nums[x] for x in range(len(nums)) if x not in [i, j]]
            if sum_diff in compare_list:
                if sorted([nums[i], nums[j], sum_diff]) not in triplets:
                    triplets.append(sorted([nums[i], nums[j], sum_diff]))
    return triplets


#x=twoSum([2,7,11,15],target=9)
x=threeSum([-1,0,1,2,-1,-4,-2,-3,3,0,4])
print(x)
'''


def firstMissingPositive(nums) -> int:
    smallest_positive = 1
    for num in sorted(nums):
        if num <= 0:
            continue
        elif num - smallest_positive == 1 or num - smallest_positive == 0:
            smallest_positive = num
        else:
            return smallest_positive+1
    return smallest_positive+1

firstMissingPositive([3,4,-1,1])

'''
def longestConsecutive(nums) -> int:
    sorted_nums = sorted(nums)
    curr_count = 1
    if len(sorted_nums) == 0:
        return 0
    for val in range(len(nums) - 1):
        if sorted_nums[val + 1] == sorted_nums[val]:
            continue
        if abs(sorted_nums[val + 1]) - abs(sorted_nums[val]) > 1:
            return curr_count
        else:
            curr_count += 1
    return curr_count


x=longestConsecutive([9,1,4,7,3,-1,0,5,8,-1,6])
print(x)
'''

'''
def maxArea(height) -> int:
    start = 0
    end = len(height) - 1
    max_area = 0
    while start < end:
        length = min(height[start],height[end])
        width = end - start
        area = length * width
        if area > max_area:
            max_area = area
            start += 1
        else:
            start += 1

        if start == end:
            end -= 1
            start = 0
    return max_area

ma=maxArea([1,8,6,2,5,4,8,3,7])
'''

'''
def characterReplacement(s: str, k: int) -> int:
    max_count = 1
    start = 0
    for c in range(len(s[start+1:]) - 1):
        if s[start] == s[start+1:][c]:
            max_count += 1
        if s[start+1:][c] == s[start+1:][c + 1]:
            max_count += 1
        else:
            start = c
    return max_count+k

c=characterReplacement("AABABBA",1)
print(c)
'''


def search(nums, target: int) -> int:
    middle = len(nums) // 2
    if target not in nums:
        return -1
    if len(nums)==2 and target != nums[-1] and target<nums[-1]:
        return 0
    elif len(nums)==2:
        return 1
    while nums[middle] != target:
        if target < nums[middle]:
            middle -= math.ceil(len(nums[:middle]) / 2)
        else:
            middle += math.ceil(len(nums[middle:]) / 2)
    return middle

c=search([-1,0,5],-1)
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
        if "linear_regression_example" == self.experiment_str:
            self.pytorch_linear_regression_example(data_exploration=False)
        elif "linear_regression_example_categorical" == self.experiment_str:
            self.pytorch_linear_regression_with_categorical(data_exploration=False)
        elif "tabular_binary_classification_example" == self.experiment_str:
            self.tabular_multiclass_classification_example(data_exploration=False)
        elif "clustering"==self.experiment_str:
            self.clustering_with_latent_embeddings_example(data_exploration=False)

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
        MLP.eval()
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


            r2 = r_squared(y_test_vals, y_pred_vals)
            print('r2 score: {}'.format(r2))

    def pytorch_linear_regression_with_categorical(self,data_exploration=False):
        Dataframes=self.dataset.load_dataset()
        if len(Dataframes)==1:
            linear_regression_categorical_dataset=Dataframes[0]
        else:
            linear_regression_categorical_dataset=pd.concat(Dataframes,ignore_index=True)

        if data_exploration:
            # doing some data exploration
            print(linear_regression_categorical_dataset.info())
            print(linear_regression_categorical_dataset.describe())
            sns.pairplot(linear_regression_categorical_dataset)
            print(linear_regression_categorical_dataset.corr())
            print(linear_regression_categorical_dataset.head())
            plt.show()

        #drop NA values
        linear_regression_categorical_dataset.dropna(inplace=True)

        '''
        linear_regression_categorical_dataset['Date_of_Journey'] = pd.to_datetime(linear_regression_categorical_dataset['Date_of_Journey'])
        linear_regression_categorical_dataset['year'] = linear_regression_categorical_dataset['Date_of_Journey'].apply(lambda x: str(x.date()).split('-')[0])
        linear_regression_categorical_dataset['month'] = linear_regression_categorical_dataset['Date_of_Journey'].apply(lambda x: str(x.date()).split('-')[1])
        linear_regression_categorical_dataset['day'] = linear_regression_categorical_dataset['Date_of_Journey'].apply(lambda x: str(x.date()).split('-')[2])

        linear_regression_categorical_dataset['year'] = linear_regression_categorical_dataset['year'].astype(int)
        linear_regression_categorical_dataset['month'] = linear_regression_categorical_dataset['month'].astype(int)
        linear_regression_categorical_dataset['day'] = linear_regression_categorical_dataset['day'].astype(int)
        '''

        linear_regression_categorical_dataset.drop(columns=['Date_of_Journey'], inplace=True)

        linear_regression_categorical_dataset['Route'] = linear_regression_categorical_dataset['Route'].apply(lambda x: "".join(x.split('→')))
        linear_regression_categorical_dataset['Total_Stops'].replace({'non-stop': '0'}, inplace=True)
        linear_regression_categorical_dataset['Total_Stops'] = linear_regression_categorical_dataset["Total_Stops"].apply(lambda x: int(x.split(" ")[0]))
        linear_regression_categorical_dataset['Total_Stops'] = linear_regression_categorical_dataset['Total_Stops'].astype(int)

        linear_regression_categorical_dataset.drop(columns=['Dep_Time','Arrival_Time'], inplace=True)
        def get_minutes(row):
            if len(row.split(" ")) == 1 and 'h' in row:
                hours_to_min = 60 * int(row[:-1])
                return hours_to_min
            elif len(row.split(" ")) == 1 and 'm' in row:
                mins = row[:-1]
                return int(mins)
            hours_and_mins = row.split('h')
            hours_to_min = 60 * int(hours_and_mins[0])
            mins = int(hours_and_mins[1][:-1])
            return int(hours_to_min + mins)

        linear_regression_categorical_dataset['Duration'] = linear_regression_categorical_dataset['Duration'].apply(lambda x: get_minutes(x))

        #set categorical columns as type category
        linear_regression_categorical_dataset['Airline']=linear_regression_categorical_dataset['Airline'].astype('category')
        linear_regression_categorical_dataset['Source']=linear_regression_categorical_dataset['Source'].astype('category')
        linear_regression_categorical_dataset['Destination']=linear_regression_categorical_dataset['Destination'].astype('category')
        linear_regression_categorical_dataset['Additional_Info']=linear_regression_categorical_dataset['Additional_Info'].astype('category')
        linear_regression_categorical_dataset['Route']=linear_regression_categorical_dataset['Route'].astype('category')

        #label encode the categorical columns
        linear_regression_categorical_dataset['Airline']=linear_regression_categorical_dataset['Airline'].cat.codes.astype('category')
        linear_regression_categorical_dataset['Source']=linear_regression_categorical_dataset['Source'].cat.codes.astype('category')
        linear_regression_categorical_dataset['Destination']=linear_regression_categorical_dataset['Destination'].cat.codes.astype('category')
        linear_regression_categorical_dataset['Additional_Info']=linear_regression_categorical_dataset['Additional_Info'].cat.codes.astype('category')
        linear_regression_categorical_dataset['Route']=linear_regression_categorical_dataset['Route'].cat.codes.astype('category')

        corr_df_int_columns=linear_regression_categorical_dataset.select_dtypes(include=['int64','float64'])
        feature_scaler = MinMaxScaler()
        #X_train = feature_scaler.fit_transform(X_train)
        #X_test = feature_scaler.transform(X_test)



        #sns.pairplot(corr_df_int_columns)
        #plt.show()

        #create features and target variables

        target=linear_regression_categorical_dataset.drop(columns=['Price'])
        features=linear_regression_categorical_dataset['Price']

        X_train, X_test, y_train, y_test = train_test_split(target, features, test_size=0.2, random_state=42)

        embedded_cols = {n: len(col.cat.categories) for n, col in X_train.items() if col.dtype.name == 'category' and len(col.cat.categories) > 2}

        #create a dataset and dataloader
        class Dataset(torch.utils.data.Dataset):
            def __init__(self, features, target, fitted_scaler_feature, fitted_scaler_target):
                self.numerical_features = features.select_dtypes(include=['int64', 'float64'])
                self.categorical_features=features.select_dtypes(include=['category'])
                self.feature_scaler = fitted_scaler_feature
                self.target_scaler = fitted_scaler_target
                self.target = target

                self.numerical_features = pd.DataFrame(self.feature_scaler.transform(self.numerical_features),
                                                       columns=self.numerical_features.columns)
                self.target = pd.DataFrame(self.target_scaler.transform(target.values.reshape(-1, 1)),
                                           columns=['Price'])

            def __len__(self):
                return len(self.numerical_features)

            def __getitem__(self, idx):
                numerical = torch.tensor(self.numerical_features.iloc[idx].values, dtype=torch.float)
                categorical = torch.tensor(self.categorical_features.iloc[idx].values, dtype=torch.int)
                target = torch.tensor(self.target.iloc[idx].values, dtype=torch.float)

                return numerical,categorical,target

        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
        fitted_scaler_feature=feature_scaler.fit(X_train.select_dtypes(include=['int64', 'float64']))
        fitted_scaler_target=target_scaler.fit(y_train.values.reshape(-1, 1))

        train_dl= torch.utils.data.DataLoader(Dataset(X_train, y_train,fitted_scaler_feature,fitted_scaler_target), batch_size=64, shuffle=True)
        test_dl= torch.utils.data.DataLoader(Dataset(X_test, y_test,fitted_scaler_feature,fitted_scaler_target), batch_size=64, shuffle=True)

        class EmbeddingNetwork(nn.Module):
            def __init__(self, categorical_dims):
                super().__init__()
                self.all_embeddings = nn.ModuleList(
                    [nn.Embedding(dim, min(50, (dim + 1) // 2)) for dim in categorical_dims])

            def forward(self, x_categorical):
                embeddings = []
                for i, e in enumerate(self.all_embeddings):
                    embeddings.append(e(x_categorical[:, i]))
                return torch.cat(embeddings, 1)


        class MLP_categorical_Regressor(nn.Module):
            def __init__(self, embedding_sizes, n_cont):
                super().__init__()
                self.embeddings = EmbeddingNetwork(embedding_sizes)
                n_emb = sum(e.embedding_dim for e in self.embeddings.all_embeddings) #length of all embeddings combined
                self.n_emb, self.n_cont = n_emb, n_cont
                self.lin1 = nn.Linear(self.n_emb + self.n_cont, 100)
                self.lin2 = nn.Linear(100, 50)
                self.lin3 = nn.Linear(50, 1)
                self.bn1 = nn.BatchNorm1d(self.n_cont)
                self.bn2 = nn.BatchNorm1d(100)
                self.bn3 = nn.BatchNorm1d(50)
                self.bn4=nn.BatchNorm1d(25)
                self.emb_drop = nn.Dropout(0.6)
                self.drops = nn.Dropout(0.3)

            def forward(self, x_numerical, x_categorical):
                x = self.embeddings(x_categorical)
                x = self.emb_drop(x)
                x2 = self.bn1(x_numerical)
                x = torch.cat([x, x2], 1)
                x = self.drops(F.relu(self.bn2(self.lin1(x))))
                x = self.drops(F.relu(self.bn3(self.lin2(x))))
                x = self.lin3(x)
                return x

        #define training loop
        def train_model(model, train_dl, loss_fn, optimizer, n_epochs,train=True):
            train_loss_per_epoch = []
            valid_loss_per_epoch = []
            for epoch in range(n_epochs):
                train_loss=[]
                model.train()
                for x_numerical,x_categorical,y in train_dl:
                    y_pred = model(x_numerical,x_categorical)
                    loss = loss_fn(y_pred,y)
                    train_loss.append(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                train_loss = sum(train_loss)
                train_loss_per_epoch.append(train_loss)
                print("Epoch ", epoch, "Train loss ", train_loss / len(train_dl))

                #model.eval()
                #with torch.no_grad():

                    #y_pred_and_orig_vals=[(model(x_numerical,x_categorical),y) for x_numerical,x_categorical,y in valid_dl]
                    #valid_loss = sum(loss_fn(model(x_numerical,x_categorical), y) for x_numerical,x_categorical,y in valid_dl)
                    #valid_loss_per_epoch.append(valid_loss.item())
                #print("Epoch ", epoch, "Validation loss ", valid_loss / len(valid_dl))
            torch.save(model.state_dict(), '/Users/sharma19/PycharmProjects/PersonalProject/models/lin_reg_categorical_model_weights.pth')
            return train_loss_per_epoch

        def validate_model(model, valid_dl, loss_fn):
            model.load_state_dict(torch.load('/Users/sharma19/PycharmProjects/PersonalProject/models/lin_reg_categorical_model_weights.pth'))
            model.eval()
            with torch.no_grad():
                y_pred_and_orig_vals=[(model(x_numerical,x_categorical),y) for x_numerical,x_categorical,y in valid_dl]
                valid_loss = sum(loss_fn(model(x_numerical,x_categorical), y) for x_numerical,x_categorical,y in valid_dl)
                print("Validation loss ", valid_loss / len(valid_dl))
            return y_pred_and_orig_vals



        #n_cont or the second argument is the number of continuous(float/int) variables present in the dataset
        MLP_model_train=MLP_categorical_Regressor(embedded_cols.values() , 2)
        MLP_model_val = MLP_categorical_Regressor(embedded_cols.values(), 2)
        train_loss=train_model(MLP_model_train, train_dl, torch.nn.MSELoss(), torch.optim.Adam(MLP_model_train.parameters(), lr=0.001),50,train=False)
        y_pred_and_orig_vals=validate_model(MLP_model_val, test_dl, torch.nn.MSELoss())

        y_pred_vals=torch.cat([y_pred.squeeze() for y_pred,y in y_pred_and_orig_vals])
        y_true_vals=torch.cat([y.squeeze() for y_pred,y in y_pred_and_orig_vals])
        print('r_squared_value is: {}'.format(r_squared(y_true_vals,y_pred_vals)))

        fig, ax = plt.subplots()
        ax.plot(train_loss, label='train loss')
        plt.show()




       # with torch.no_grad():
           # y_pred_vals=[]
            #for x_test, y_test in test_loader:
                #y_pred = MLP(x_test)
                #loss = criterion(y_pred, y_test)
                #y_pred_vals.append(y_pred.squeeze())
                #print('loss on test set: {}'.format(loss.item()))

            #y_pred_vals=torch.cat(y_pred_vals)
            #y_test_vals=torch.cat([y_test for x_test, y_test in test_loader]).squeeze()


    def tabular_multiclass_classification_example(self,data_exploration=True):
        Dataframes=self.dataset.load_dataset()
        if len(Dataframes)==1:
            tabular_classification_dataset=Dataframes[0]
        else:
            tabular_classification_dataset=pd.concat(Dataframes,ignore_index=True)

        if data_exploration:
            print(tabular_classification_dataset.info())
            print(tabular_classification_dataset.describe())

        #drop columns with histogram in column name
        tabular_classification_dataset=tabular_classification_dataset.drop([col for col in tabular_classification_dataset.columns if 'histogram' in col],axis=1)
        print(tabular_classification_dataset.info())

        if data_exploration:
            #plot histograms
            for col in tabular_classification_dataset.columns:
                tabular_classification_dataset.hist(col,bins=50, figsize=(20, 15))
                plt.title(col)
                plt.show()
            #corr matrix
            corr_matrix=tabular_classification_dataset.corr()
            #plot heatmap
            plt.figure(figsize=(20, 15))
            sns.heatmap(corr_matrix, annot=True)
            plt.show()

        #drop columns with no correlation with target

        #tabular_classification_dataset.drop(['accelerations','uterine_contractions','mean_value_of_short_term_variability','mean_value_of_long_term_variability'],axis=1,inplace=True)

        #create target and features
        target=tabular_classification_dataset['fetal_health']
        features=tabular_classification_dataset.drop('fetal_health',axis=1)
        target.replace({1:0,2:1,3:2},inplace=True) #need to do this as loss function expects target to be in range 0 to n_classes-1
        #plot bar chart of target
        target.value_counts().plot(kind='bar')
        #add percentage labels to bar chart
        for p in plt.gca().patches:
            percentage = '{:.1f}%'.format(100 * p.get_height() / len(target))
            x = p.get_x() + p.get_width() / 2 - 0.05
            y = p.get_y() + p.get_height()
            plt.annotate(percentage, (x, y), size=12)
        plt.title('Target distribution before SMOTE')
        plt.show()

        #Use SMOTE to balance the dataset because the data is imbalanced
        resample_smote = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))
        features, target = resample_smote.fit_resample(features, target)

        target.value_counts().plot(kind='bar')
        #add percentage labels to bar chart
        for p in plt.gca().patches:
            percentage = '{:.1f}%'.format(100 * p.get_height() / len(target))
            x = p.get_x() + p.get_width() / 2 - 0.05
            y = p.get_y() + p.get_height()
            plt.annotate(percentage, (x, y), size=12)
        plt.title('Target distribution after SMOTE')
        plt.show()


        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        #create a simple dataset class and dataloader to load the data

        class_counts=target.value_counts()
        class_weights=[1/class_counts[i] for i in range(len(class_counts))] #inverse of count is the weight
        #num_samples=len(class_counts)*class_counts[0]
        #sampler=WeightedRandomSampler(weights=class_weights,num_samples=4965,replacement=True)

        class classification_dataset(torch.utils.data.Dataset):
            def __init__(self, features, target):
                self.features = features
                self.target = target

            def __len__(self):
                return self.features.shape[0]

            def __getitem__(self, idx):
                return self.features[idx], self.target[idx]

        train_dataset=classification_dataset(torch.tensor(X_train.values),torch.tensor(y_train.values))
        test_dataset=classification_dataset(torch.tensor(X_test.values),torch.tensor(y_test.values))

        #create data loaders
        train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

        #create a simple MLP model
        class MLP_categorical_classifier(torch.nn.Module):
            def __init__(self):
                super(MLP_categorical_classifier, self).__init__()
                self.fc1 = torch.nn.Linear(11,64)
                self.relu = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(64, 32)
                self.fc3 = torch.nn.Linear(32, 16)
                self.fc4 = torch.nn.Linear(16, 3)

            def forward(self, x):
                x = self.fc1(x.to(torch.float32)) #notice how we have to convert the input to float32 as the wight matrix is float32 dtype and our data was float64,
                                                      # this would throw an error if we didn't cast the input to float32
                x = self.relu(x)
                x = self.fc2(x)
                x = self.relu(x)
                x = self.fc3(x)
                x = self.relu(x)
                output = self.fc4(x)
                return output

        #create training loop
        writer = SummaryWriter('runs/fetal_health_classifier') #intialize tensorboard writer
        # to run tensorboard, open a terminal and type: tensorboard --logdir=runs in the terminal
        # visit http://localhost:6006 to view the graphs
        learning_rate = 0.01
        epochs = 200
        MLP = MLP_categorical_classifier()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(MLP.parameters(), lr=learning_rate) # notice how the optimizer has changed, there are many kinds of optimizers explained in the documentation


        #train the model
        loss_per_epoch = []
        for epoch in range(epochs):
            correct = 0
            total = 0
            #loop = tqdm(enumerate(train_dl), total=len(train_dl), leave=False)
            for x_train, y_train in train_dl:
                optimizer.zero_grad()
                y_pred = MLP(x_train)
                loss = criterion(y_pred, y_train.to(torch.int64))
                writer.add_scalar('Loss/train', loss, epoch) #visit http://localhost:6006 to view the graphs
                loss.backward()
                optimizer.step()

                # Calculate accuracy
                _, predicted = torch.max(y_pred.data, 1)
                total += y_train.size(0)
                correct += (predicted == y_train).sum().item()
            acc=100 * correct / total
            writer.add_scalar('Accuracy/train', acc, epoch) #visit http://localhost:6006 to view the graphs

            loss_per_epoch.append(loss.item())
            #loop.set_description(f"Epoch [{epoch}/{epochs}]")
            #loop.set_postfix(loss=loss.item(), acc=acc)
            print('epoch {}, loss {}, accuracy {}'.format(epoch, loss.item(),acc))

        print('training finished')
        #plot loss
        #plt.plot(loss_per_epoch)
        #plt.title('loss')
        #plt.show()

        #test the model
        correct = 0
        total = 0
        original_labels=[]
        predicted_labels=[]
        predicted_logits=[]
        MLP.eval()
        with torch.no_grad():
            for x_test, y_test in test_dl:
                y_pred = MLP(x_test)
                #get the predicted logit and get the index of the max logit, this is the predicted class
                _, predicted = torch.max(y_pred.data, 1)
                total += y_test.size(0)
                correct += (predicted == y_test).sum().item()
                original_labels.append(y_test.data)
                predicted_labels.append(predicted)
                predicted_logits.append(_)
        original_labels=torch.cat(original_labels,0)
        predicted_labels=torch.cat(predicted_labels,0)

        #for multiclass problems either you calculate all the below scores for each target class or you caclulate it and then
        #average it over all the classes, this is what average='macro' does
        F1_score=MulticlassF1Score(num_classes=3,average='macro')
        cohen_kappa_score=MulticlassCohenKappa(num_classes=3)
        precision=MulticlassPrecision(num_classes=3,average='macro')
        recall=MulticlassRecall(num_classes=3,average='macro')

        print('accuracy on test set: {} %'.format(100 * correct / total))
        print(f'F1 score: {F1_score(predicted_labels,original_labels).item()}')
        print(f'Cohen Kappa score: {cohen_kappa_score(predicted_labels,original_labels).item()}')
        print(f'Precision is: {precision(predicted_labels,original_labels).item()} and Recall is: {recall(predicted_labels,original_labels).item()}')
        print('Example completed')


    def clustering_with_latent_embeddings_example(self,data_exploration=True):
        '''
        this example shows how to use the latent embeddings from an autoencoder to cluster the data
        Implementing a Deep Clustering Network (please thoroughly read the paper to understand the code) - https://arxiv.org/pdf/1610.04794.pdf
        the code is based on this repo - https://github.com/xuyxu/Deep-Clustering-Network/tree/master
        The objective of this network is to simultaneously learn the latent vector and optimise it for K-means clustering while also implementing K-means to cluster the latent vectors
        it uses an autoencoder for dimensionality reduction and K- means to cluster the latent vectors
        the loss function is the sum of the reconstruction loss and the clustering loss
        the reconstruction loss used is  MSE, the K-means loss is implemented with a regularization term
        '''

        Dataframes=self.dataset.load_dataset()
        if len(Dataframes)==1:
            anime_clustering_dataset=Dataframes[0]
        else:
            anime_clustering_dataset=pd.concat(Dataframes,ignore_index=True)

        if data_exploration:
            print(anime_clustering_dataset.head())
            print(anime_clustering_dataset.info())
            print(anime_clustering_dataset.describe())
            print(anime_clustering_dataset.isna().sum())

        #drop na values
        anime_clustering_dataset.dropna(inplace=True)

        #tokenize the name and genre columns
        tokenizer = RegexpTokenizer(r'\w+')
        anime_clustering_dataset['name']=anime_clustering_dataset['name'].apply(lambda x: ' '.join(tokenizer.tokenize(x)))
        anime_clustering_dataset['genre']=anime_clustering_dataset['genre'].apply(lambda x: ' '.join(tokenizer.tokenize(x)))
        #replace spaces with underscores in name column
        #anime_clustering_dataset['name']=anime_clustering_dataset['name'].apply(lambda x: x.replace(' ','_'))
        anime_clustering_dataset['genre'] = anime_clustering_dataset['genre'].apply(lambda x: x.replace(' ', '_'))
        #write name column to text file
        columns_to_save=['name','genre']
        for column in columns_to_save:
            column_to_save = anime_clustering_dataset[column]

            # Define the output file path
            output_file_path = 'data/anime_{}.txt'.format(column)

            # Write the column values to the .txt file for possible future embeddings via fasttext
            with open(output_file_path, 'w') as f:
                for value in column_to_save:
                    f.write(value + '\n')

        #anime_clustering_dataset['name']= anime_clustering_dataset.name.astype('category')
        #anime_clustering_dataset['genre']= anime_clustering_dataset.genre.astype('category')
        anime_clustering_dataset['type']=anime_clustering_dataset.type.astype('category')

        def get_fasttext_embeddings():
            files_in_directory = [os.path.join('data', file) for file in os.listdir('data') if '.txt' in file]
            model_list= {}
            for file in files_in_directory:
                model = fasttext.train_unsupervised(file, model='skipgram')

                model_list[file.split("/")[-1]]=model
            return model_list
        models=get_fasttext_embeddings()
        embedding_sizes=[model.get_dimension() for model in models.values()]
        cat_embeddings_name = [models["anime_name.txt"].get_word_vector(word) for word in anime_clustering_dataset['name']]
        cat_embeddings_name = torch.tensor(np.array(cat_embeddings_name), dtype=torch.float32)
        #visualize the embeddings




        cat_embeddings_genre = [models["anime_genre.txt"].get_word_vector(word) for word in anime_clustering_dataset['genre']]
        cat_embeddings_genre = torch.tensor(np.array(cat_embeddings_genre), dtype=torch.float32)

        anime_clustering_dataset['genre']=cat_embeddings_genre
        anime_clustering_dataset['name'] = cat_embeddings_name
        anime_clustering_dataset['type']=anime_clustering_dataset.type.cat.codes.astype('float64')



        #anime_clustering_dataset['name']=anime_clustering_dataset.name.cat.codes.astype('category')
        #anime_clustering_dataset['genre']=anime_clustering_dataset.genre.cat.codes.astype('category')

        corr_df_int_columns=anime_clustering_dataset.select_dtypes(include=['int64','float64'])
        feature_scaler_train = MinMaxScaler()
        feature_scaler_test = MinMaxScaler()
        train_set=anime_clustering_dataset.sample(frac=0.8,random_state=2)
        test_set=anime_clustering_dataset.drop(train_set.index)
        train_set_numeric=train_set.select_dtypes(include=['int64','float64'])
        test_set_numeric=test_set.select_dtypes(include=['int64','float64'])
        train_set_categorical=train_set.select_dtypes(include=['float32'])
        test_set_categorical=test_set.select_dtypes(include=['float32'])

        #embedded_cols = {n: len(col.cat.categories) for n, col in train_set.items() if col.dtype.name == 'category' and len(col.cat.categories) > 2}

        '''
        def get_fasttext_embeddings():
            files_in_directory = [os.path.join('data', file) for file in os.listdir('data') if '.txt' in file]
            model_list= []
            for file in files_in_directory:
                model = fasttext.train_unsupervised(file, model='skipgram')

                model_list.append(model)
            return model_list
        '''


        class Autoencoder_Dataset(torch.utils.data.Dataset):
            def __init__(self, num_features, cat_features):
                self.num_features = num_features
                self.cat_features=cat_features

            def __len__(self):
                return len(self.num_features)

            def __getitem__(self, idx):
                numerical = torch.tensor(self.num_features.values).float()[idx]
                #categorical = torch.tensor(self.cat_features.values,requires_grad=True)[idx]
                #categorical=[torch.tensor(i[idx],requires_grad=True) for i in self.cat_features]
                categorical=[i[idx].clone().detach().requires_grad_(True) for i in self.cat_features]

                return F.normalize(numerical.unsqueeze(0)).squeeze(0),categorical[0],categorical[1]

        train_dl= torch.utils.data.DataLoader(Autoencoder_Dataset(train_set_numeric,[cat_embeddings_name,cat_embeddings_genre]), batch_size=64, shuffle=True)
        test_dl= torch.utils.data.DataLoader(Autoencoder_Dataset(test_set_numeric,[cat_embeddings_name,cat_embeddings_genre]), batch_size=64, shuffle=True)


        '''
        class EmbeddingNetwork(nn.Module):
            def __init__(self, categorical_dims,use_pretrained_embeddings=True,model_list=None):
                super().__init__()
                self.categorical_dims= categorical_dims
                self.use_pretrained_embeddings=use_pretrained_embeddings
                if not use_pretrained_embeddings:
                    self.all_embeddings = nn.ModuleList(
                        [nn.Embedding(dim, min(50, (dim + 1) // 2)) for dim in self.categorical_dims])
                else:
                    # Create embedding layer
                    self.all_embeddings=nn.ModuleList()
                    for model in model_list:
                        embedding_dim = model.get_dimension()
                        embedding_layer = [nn.Embedding(num_embeddings=5, embedding_dim=embedding_dim)]
                        self.all_embeddings.append(embedding_layer)

            def forward(self, x_categorical):
                if not self.use_pretrained_embeddings:
                    embeddings = []
                    for i, e in enumerate(self.all_embeddings):
                        embeddings.append(e(x_categorical[:, i]))
                    return F.normalize(torch.cat(embeddings, 1))
                else:
                    embeddings=[]
                    for i,e in enumerate(self.all_embeddings):
                        embeddings.append(e(x_categorical[:,i]))
                        torch.cat(embeddings, 1)   
        '''
        class EmbeddingNetwork(nn.Module):
            def __init__(self, categorical_dims):
                super().__init__()
                self.categorical_dims= categorical_dims
                self.all_embeddings = nn.ModuleList(
                    [nn.Embedding(dim, min(50, (dim + 1) // 2)) for dim in self.categorical_dims])
                self.files_in_directory = [os.path.join('data',file) for file in os.listdir('data') if '.txt' in file]

            def forward(self, x_categorical):
                embeddings = []
                for i, e in enumerate(self.all_embeddings):
                    embeddings.append(e(x_categorical[:, i]))
                return F.normalize(torch.cat(embeddings, 1))




        #emb_out=[]
        #num_out=[]
        #embeddings = EmbeddingNetwork(embedded_cols.values(),use_pretrained_embeddings=True,model_list=get_fasttext_embeddings())
        #for numerical_feat,categorical_feat in  train_dl:
            #embeddings_output=embeddings(categorical_feat)
            #emb_out.append(embeddings_output)
            #num_out.append(numerical_feat)
        #print('done')

        #for sake of simplicity adding batch k-means class code here

        def _parallel_compute_distance(X, cluster):
            n_samples = X.shape[0]
            dis_mat = np.zeros((n_samples, 1))
            #calculating euclidean distance of points to cluster centers
            for i in range(n_samples):
                dis_mat[i] += np.sqrt(np.sum((X[i] - cluster) ** 2, axis=0))
            return dis_mat

        class batch_KMeans:

            def __init__(self, latent_dim, n_clusters,n_jobs=1):
                self.latent_dim = latent_dim
                self.n_clusters = n_clusters
                self.clusters = np.zeros((self.n_clusters, self.latent_dim))
                self.count = 100 * np.ones((self.n_clusters))  # serve as learning rate
                self.n_jobs = n_jobs

            def _compute_dist(self, X):
                dis_mat = Parallel(n_jobs=self.n_jobs)(
                    delayed(_parallel_compute_distance)(X, self.clusters[i])
                    for i in range(self.n_clusters))
                dis_mat = np.hstack(dis_mat)

                return dis_mat

            def init_cluster(self, X, indices=None):
                """ Generate initial clusters using sklearn.Kmeans """
                model = KMeans(n_clusters=self.n_clusters,
                               n_init=20)
                model.fit(X)
                self.clusters = model.cluster_centers_  # copy clusters

            def update_cluster(self, X, cluster_idx):
                """ Update clusters in Kmeans on a batch of data """
                n_samples = X.shape[0]
                for i in range(n_samples):
                    self.count[cluster_idx] += 1
                    eta = 1.0 / self.count[cluster_idx]
                    updated_cluster = (torch.from_numpy((1 - eta) * self.clusters[cluster_idx])+
                                       eta * X[i])
                    self.clusters[cluster_idx] = updated_cluster

            def update_assign(self, X):
                """ Assign samples in `X` to clusters """
                dis_mat = self._compute_dist(X)

                return np.argmin(dis_mat, axis=1)
        

        class Autoencoder(nn.Module):
            def __init__(self, embedding_size, n_continuous_features):
                super().__init__()
                #self.embeddings=EmbeddingNetwork(embedding_sizes)
                #n_emb = sum(e.embedding_dim for e in self.embeddings.all_embeddings) #length of all embeddings combined
                n_emb=sum(embedding_size)
                self.n_emb, self.n_cont = n_emb, n_continuous_features
                self.hidden_dim=self.n_emb + self.n_cont

                def init_weights(m):
                    if isinstance(m, nn.Linear):
                        torch.nn.init.xavier_uniform_(m.weight)
                        m.bias.data.fill_(0.01)


                #nn.Sequential is a container for several layers that can be stacked together it implements the forward pass automatically
                self.encoder = nn.Sequential(
                    nn.Linear(self.hidden_dim, 200),
                    nn.ReLU(),
                    nn.Linear(200, 100),
                    nn.ReLU(),
                    nn.Linear(100, 50),
                    nn.ReLU(),
                    nn.Linear(50, 16),
                    nn.ReLU(),
                    nn.Linear(16, 8),
                    nn.ReLU(),
                    nn.Linear(8, 2)
                    )
                self.encoder.apply(init_weights)
                self.decoder = nn.Sequential(
                    nn.Linear(2, 8),
                    nn.ReLU(),
                    nn.Linear(8, 16),
                    nn.ReLU(),
                    nn.Linear(16, 50),
                    nn.ReLU(),
                    nn.Linear(50, 100),
                    nn.ReLU(),
                    nn.Linear(100, 200),
                    nn.ReLU(),
                    nn.Linear(200, self.hidden_dim))
                self.decoder.apply(init_weights)

            def forward(self, x_categorical_feat_1,x_categorical_feat_2,x_numerical,latent=False):

                #x = self.embeddings(x_categorical)
                x = torch.cat([x_categorical_feat_1,x_categorical_feat_2, x_numerical], 1).to(torch.float32)
                x = self.encoder(x)
                if latent:
                    return x
                x = self.decoder(x)
                #sig = nn.Sigmoid()
                return x

        def generate_target(x_categorical_feat_1,x_categorical_feat_2,x_numerical):
            #embedding_network=EmbeddingNetwork(embedded_cols.values())
            #embedded_values=embedding_network(x_categorical)
            #x = torch.cat([x_numerical,embedded_values], 1)
            #no need for the gradient for the target hence we detach it
            #return x.detach()
            return torch.cat([x_categorical_feat_1,x_categorical_feat_2, x_numerical],1).detach()



        def generate_KMeans_target(x_categorical_feat_1,x_categorical_feat_2,x_numerical):
            X=torch.cat([x_categorical_feat_1,x_categorical_feat_2,x_numerical],1).detach().numpy()
            kmeans=KMeans(n_clusters=10, n_init=20)
            labels=kmeans.fit_predict(X)
            return labels

        train=True
        #train the autoencoder
        #autoencoder = Autoencoder(embedded_cols.values(), len(corr_df_int_columns.columns))
        autoencoder = Autoencoder(embedding_sizes, len(corr_df_int_columns.columns))
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
        writer = SummaryWriter('runs/anime_autoencoder')
        if train:
            epochs=50
            for epoch in range(epochs):
                train_loss = 0.0
                torch.manual_seed(42)
                np.random.seed(42)
                random.seed(42)
                for x_numerical,x_categorical_feat_1,x_categorical_feat_2 in train_dl:
                    optimizer.zero_grad()
                    output = autoencoder(x_categorical_feat_1,x_categorical_feat_2,x_numerical)
                    target_train=generate_target(x_categorical_feat_1,x_categorical_feat_2,x_numerical)
                    loss = criterion(output, target_train)
                    writer.add_scalar('Loss/train', loss, epoch)  # visit http://localhost:6006 to view the graphs
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                train_loss = train_loss / len(train_dl)
                print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))
            torch.save(autoencoder.state_dict(),
                       '/Users/sharma19/PycharmProjects/PersonalProject/models/anime_autoencoder_model_weights.pth')
        else:
            #validate the autoencoder
            autoencoder.load_state_dict(torch.load('/Users/sharma19/PycharmProjects/PersonalProject/models/anime_autoencoder_model_weights.pth'))
            autoencoder.eval()
            test_loss = 0.0
            target_vals=[]
            predicted_vals=[]
            with torch.no_grad():
                torch.manual_seed(42)
                for x_numerical,x_categorical_feat_1,x_categorical_feat_2 in test_dl:
                    output = autoencoder(x_categorical_feat_1,x_categorical_feat_2,x_numerical)
                    predicted_vals.append(output)
                    target_val=generate_target(x_categorical_feat_1,x_categorical_feat_2,x_numerical)
                    target_vals.append(target_val)
                    loss = criterion(output, target_val)
                    test_loss += loss.item()
                test_loss = test_loss / len(test_dl)
                print('Validation Loss: {:.6f}'.format(test_loss))
            plt.scatter(torch.cat([x for x in torch.cat([t for t in target_vals])])[:10], torch.cat([x for x in torch.cat(predicted_vals)])[:10])
            plt.xlabel('target')
            plt.ylabel('decoder output')
            plt.title('Target vs Decoder output')
            plt.show()

            #calculate the r2 score


            r2 = r_squared(torch.cat([t for t in target_vals]), torch.cat(predicted_vals))
            print('r2 score: {}'.format(r2))

        class DCN(nn.Module):
            def __init__(self,lambda_coef,beta_coef,train_loader,test_loader,embedding_sizes,continous_features):
                super().__init__()
                self.autoencoder = Autoencoder(embedding_sizes, len(continous_features.columns))
                self.kmeans = batch_KMeans(latent_dim=2,n_clusters=10)
                self.lambda_coef=lambda_coef
                self.beta_coef=beta_coef
                self.train_loader=train_loader
                self.test_loader=test_loader
                self.criterion = nn.MSELoss()
                self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

                if not self.beta_coef > 0:
                    msg = 'beta should be greater than 0 but got value = {}.'
                    raise ValueError(msg.format(self.beta))

                if not self.lambda_coef > 0:
                    msg = 'lamda should be greater than 0 but got value = {}.'
                    raise ValueError(msg.format(self.lamda))

            def combined_loss(self,x_categorical_feat_1,x_categorical_feat_2,x_numerical,cluster_id):
                batch_size=x_categorical_feat_1.shape[0]
                self.autoencoder.load_state_dict(torch.load('/Users/sharma19/PycharmProjects/PersonalProject/models/anime_autoencoder_model_weights.pth'))
                reconstructed_output= self.autoencoder(x_categorical_feat_1,x_categorical_feat_2,x_numerical)
                latent_vector= self.autoencoder(x_categorical_feat_1,x_categorical_feat_2,x_numerical,latent=True)
                X=generate_target(x_categorical_feat_1,x_categorical_feat_2,x_numerical)
                rec_loss=self.lambda_coef*self.criterion(X,reconstructed_output)


                #clustering_loss

                dist_loss = torch.tensor(0.)
                clusters = torch.FloatTensor(self.kmeans.clusters)
                for i in range(batch_size):
                    diff_vec = latent_vector[i] - clusters[cluster_id[i]]
                    # basically taking the dot product below can be done via torch.dot(diff_vec,diff_vec.T) also
                    sample_dist_loss = torch.matmul(diff_vec.view(1, -1),
                                                    diff_vec.view(-1, 1))
                    dist_loss += 0.5 * self.beta_coef * torch.squeeze(sample_dist_loss)

                return (rec_loss + dist_loss,
                        rec_loss.detach().numpy(),
                        dist_loss.detach().numpy())

            #Dont need to pretrain because we are using pretrained autoencoder by loading the weights

            '''
            def pretrain(self, train_loader, epoch=100, verbose=True):

                if verbose:
                    print('========== Start pretraining ==========')

                rec_loss_list = []

                self.train()
                for e in range(epoch):
                    for batch_idx, (data, _) in enumerate(train_loader):
                        batch_size = data.size()[0]
                        data = data.to(self.device).view(batch_size, -1)
                        rec_X = self.autoencoder(data)
                        loss = self.criterion(data, rec_X)

                        if verbose and (batch_idx + 1) % self.args.log_interval == 0:
                            msg = 'Epoch: {:02d} | Batch: {:03d} | Rec-Loss: {:.3f}'
                            print(msg.format(e, batch_idx + 1,
                                             loss.detach().cpu().numpy()))
                            rec_loss_list.append(loss.detach().cpu().numpy())

                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                self.eval()

                if verbose:
                    print('========== End pretraining ==========\n')
            '''
            def initialise_clusters(self):
                batch_X = []
                for x_numerical,x_categorical_feat_1,x_categorical_feat_2 in self.train_loader:
                    batch_size = x_categorical_feat_1.size()[0]
                    latent_X = self.autoencoder(x_categorical_feat_1,x_categorical_feat_2,x_numerical, latent=True)
                    batch_X.append(latent_X.detach().cpu().numpy())
                batch_X = np.vstack(batch_X)
                self.kmeans.init_cluster(batch_X)

            def fit(self,epoch,train_loader):
                self.train()
                rec_loss_list = []
                dist_loss_list = []

                # epochs will be defined in the main/solver function
                #for e in range(epoch):
                for batch_idx, (x_numerical,x_categorical_feat_1,x_categorical_feat_2) in enumerate(train_loader):

                    batch_size = x_numerical.size()[0]
                    #data = data.view(batch_size, -1)

                    with torch.no_grad():
                        latent_X = self.autoencoder(x_categorical_feat_1,x_categorical_feat_2,x_numerical, latent=True)

                    cluster_id = self.kmeans.update_assign(latent_X.cpu().numpy())
                    # [Step-2] Update clusters in batch Clustering
                    elem_count = np.bincount(cluster_id,
                                             minlength=self.kmeans.n_clusters)
                    for k in range(self.kmeans.n_clusters):
                        # avoid empty slicing
                        if elem_count[k] == 0:
                            continue
                        #latent_X[cluster_id == k] gets all of elements in a particular cluster id
                        self.kmeans.update_cluster(latent_X[cluster_id == k], k)


                    loss, rec_loss, dist_loss = self.combined_loss(x_categorical_feat_1, x_categorical_feat_2,x_numerical, cluster_id)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    msg = 'Epoch: {:02d} | Batch: {:03d} | Loss: {:.3f} | Rec-' \
                          'Loss: {:.3f} | Dist-Loss: {:.3f}'
                    print(msg.format(epoch, batch_idx + 1,
                                     loss.detach().cpu().numpy(),
                                     rec_loss, dist_loss))

        def evaluate(model, test_loader):
            y_test = []
            y_pred = []
            for x_numerical,x_categorical_feat_1,x_categorical_feat_2 in test_loader:

                batch_size = x_numerical.size()[0]
                #data = data.view(batch_size, -1).to(model.device)
                latent_X = model.autoencoder(x_numerical,x_categorical_feat_1,x_categorical_feat_2, latent=True)

                latent_X = latent_X.detach().cpu().numpy()
                k_target=generate_KMeans_target(x_categorical_feat_1,x_categorical_feat_2,x_numerical)

                target=generate_target(x_categorical_feat_1,x_categorical_feat_2,x_numerical)

                y_test.append(target.view(-1, 1).numpy())
                cluster_predictions=model.kmeans.update_assign(latent_X).reshape(-1, 1)
                y_pred.append(cluster_predictions)
                #get the silhouette scores for the clusters

                sil_score=silhouette_score(latent_X,cluster_predictions)
                #silhouette_samples()
                #visualize the clusters
                plt.scatter(latent_X[:,0],latent_X[:,1],c=cluster_predictions)
                plt.title('Silhouette score: {}'.format(sil_score))
                plt.show()

            y_test = np.vstack(y_test).reshape(-1)
            y_pred = np.vstack(y_pred).reshape(-1)
            return (normalized_mutual_info_score(y_test, y_pred),
                    adjusted_rand_score(y_test, y_pred))

        def solver(epoch, model, train_loader, test_loader):

            #rec_loss_list = model.pretrain(train_loader, epoch=args.pre_epoch)
            model.initialise_clusters()
            nmi_list = []
            ari_list = []

            for e in range(epoch):
                model.train()
                model.fit(e, train_loader)

                model.eval()
                NMI, ARI = evaluate(model, test_loader)  # evaluation on the test_loader
                nmi_list.append(NMI)
                ari_list.append(ARI)

                print('Epoch: {:02d} | NMI: {:.3f} | ARI: {:.3f}'.format(
                    e + 1, NMI, ARI))

            return nmi_list, ari_list

        solver(10,DCN(0.1,0.1,train_dl,test_dl,embedding_sizes,corr_df_int_columns),train_dl,test_dl)
        print('Autoencoder Trained')















































