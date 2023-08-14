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
import torch.optim as optim
import datetime
from torch.utils.data import WeightedRandomSampler
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils import r_squared
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassF1Score,MulticlassCohenKappa,MulticlassAUROC


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
            def __init__(self, features, target,fitted_scaler_train,fitted_scaler_target):
                self.numerical_features = features.select_dtypes(include=['int64', 'float64'])
                self.categorical_features=features.select_dtypes(include=['category'])
                self.feature_scaler = fitted_scaler_train
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
        fitted_scaler_train=feature_scaler.fit(X_train.select_dtypes(include=['int64', 'float64']))
        fitted_scaler_target=target_scaler.fit(y_train.values.reshape(-1, 1))

        train_dl= torch.utils.data.DataLoader(Dataset(X_train, y_train,fitted_scaler_train,fitted_scaler_target), batch_size=64, shuffle=True)
        test_dl= torch.utils.data.DataLoader(Dataset(X_test, y_test,fitted_scaler_train,fitted_scaler_target), batch_size=64, shuffle=True)

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

        F1_score=MulticlassF1Score(num_classes=3,average='macro')
        cohen_kappa_score=MulticlassCohenKappa(num_classes=3)

        print('accuracy on test set: {} %'.format(100 * correct / total))
        print(f'F1 score: {F1_score(predicted_labels,original_labels).item()}')
        print(f'Cohen Kappa score: {cohen_kappa_score(predicted_labels,original_labels).item()}')




































