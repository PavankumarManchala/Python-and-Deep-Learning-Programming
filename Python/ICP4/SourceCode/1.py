import pandas as pd

train_df = pd.read_csv('train.csv')
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
train_df['Sex'] = train_df['Sex'].map({'female': 1, 'male': 0}).astype(int)
print(train_df['Survived'].corr(train_df['Sex']))