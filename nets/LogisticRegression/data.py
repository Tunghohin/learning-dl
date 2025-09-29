import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils.dataset import download_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class TitanicDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        features = row.drop('Survived').values.astype('float32')
        label = row['Survived']

        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return features, label

def load_data():
    download_dataset('titanic')
    df = pd.read_csv('./datasets/titanic/tested.csv')

    df = df.drop(columns=['PassengerId', 'Name', 'Cabin', 'Ticket'])
    df['Sex'] = df['Sex'].map({"male": 1, "female": 0})
    df['Embarked'] = df['Embarked'].map({"S": 0, "C": 1, "Q": 2})
    df = pd.get_dummies(df, columns=["Embarked"], prefix="Embarked")
    df = pd.get_dummies(df, columns=["Pclass"], prefix="Pclass")
    df = pd.get_dummies(df, columns=["SibSp"], prefix="SibSp")

    for col in df.columns:
        df.fillna({col: df[col].median()}, inplace=True)

    scalar = StandardScaler()
    df[['Age', "Fare"]] = scalar.fit_transform(df[['Age', "Fare"]])

    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
   
    train_set = TitanicDataset(train_set.reset_index(drop=True))
    test_set = TitanicDataset(test_set.reset_index(drop=True))

    return train_set, test_set

def prepare_data(dataset, batch_size=64):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)