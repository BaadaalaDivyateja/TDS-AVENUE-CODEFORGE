import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
gender_submission = pd.read_csv('gender_submission.csv')

# Preprocessing data
def preprocess_data(df):
    df = df.drop(['Cabin', 'Ticket', 'Name'], axis=1)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    return df

# Preprocess the data
train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# Define features and target variable
X = train_data.drop('Survived', axis=1)
y = train_data['Survived']

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_val)
print('Logistic Regression Model Accuracy:', accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))

# Visualize some data
sns.countplot(x='Survived', data=train_data)
plt.show()

# Compare with gender submission baseline model
gender_baseline_accuracy = accuracy_score(y_val, gender_submission['Survived'][:len(y_val)])
print('Gender Submission Baseline Model Accuracy:', gender_baseline_accuracy)
