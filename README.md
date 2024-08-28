# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Design and implement a neural network regression model to accurately predict a continuous target variable based on a set of input features within the provided dataset. The objective is to develop a robust and reliable predictive model that can capture complex relationships in the data, ultimately yielding accurate and precise predictions of the target variable. The model should be trained, validated, and tested to ensure its generalization capabilities on unseen data, with an emphasis on optimizing performance metrics such as mean squared error or mean absolute error. This regression model aims to provide valuable insights into the underlying patterns and trends within the dataset, facilitating enhanced decision-making and understanding of the target variable's behavior.

## Neural Network Model

![image](https://github.com/user-attachments/assets/f472d537-9f4e-485c-9672-9d987a2b7812)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: A.J.PRANAV
### Register Number: 212222230107
```
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

auth.authenticate_user()
creds, _ = default()

gc = gspread.authorize(creds)

worksheet = gc.open("Data").sheet1

data = worksheet.get_all_values()

df = pd.DataFrame(data[1:],columns=data[0])

#df.columns = df.iloc[0]

#df = df.iloc[1:]

print(df)

df = df.rename(columns={'input': 'Input','output': 'Output'})

df = df.astype({'Input':'float'})
df = df.astype({'Output':'float'})

X = df[['Input']].values
y = df['Output'].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=33)

Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train = Scaler.transform(X_train)

model = Sequential([
    Dense(units=9, activation='relu',input_shape=[1]),
    Dense(units=9, activation='relu'),
    Dense(units=9, activation='relu'),
    Dense(units=1)
])

model.compile(optimizer='rmsprop',loss='mse')

model.fit(X_train,y_train,epochs=2000)

X_test1=Scaler.transform(X_test)
model.evaluate(X_test1,y_test)

pd.DataFrame(model.summary())

X_n1=[[20]]
X_n1=Scaler.transform(X_n1)
model.predict(X_n1)

```
## Dataset Information

![image](https://github.com/user-attachments/assets/cbf1cc82-8f18-4562-9bd3-b57009819b91) 


## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/895a1455-532e-476e-a640-d0e706d2fc52)


### Test Data Root Mean Squared Error

![image](https://github.com/user-attachments/assets/87403e21-67fe-4e34-a3f9-a1eca44b13a3)


### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/84466df2-7139-40fb-a4c0-308c8589632a)


## RESULT
A neural network regression model for the given dataset has been developed sucessfully.
