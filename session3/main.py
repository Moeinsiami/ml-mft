import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

dataset = pd.read_csv("50_Startups.csv")
x = dataset.iloc[:, :-1].values
y = dataset["Profit"].values.reshape(-1, 1)
# y = dataset["Profit"].values.reshape(2, 25)

ct = ColumnTransformer(transformers=[
    ('encoder', OneHotEncoder(), [3])
    ]
    , remainder="passthrough")
x = np.array(ct.fit_transform(x))
# print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_train)

print(len(y_pred))

y_pred_reshape = y_pred.reshape(len(y_pred), 1)
y_test_reshape = y_test.reshape(len(y_test), 1)

print(y_pred_reshape)

plt.figure(figsize=(10, 5))
plt.plot(y_pred, label="predictions", marker='o', linestyle='--')



plt.plot(y_test_reshape, label="Value", marker='x')
plt.legend()
plt.grid(True)
plt.show()

# acc = r2_score(y_train, y_pred)
# print(acc)
#
# acc2 = r2_score(y_test, model.predict(x_test))
# print(acc2)