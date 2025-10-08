import matplotlib.pyplot

import numpy as np
import pandas as pd
df = pd.read_csv("week2.csv")
print(df.head())
X1=df.iloc[:,0]
X2=df.iloc[:,1]
X=np.column_stack((X1,X2))
y=df.iloc[:,2]

from sklearn.linear_model import LogisticRegression
model = LogisticRegression().fit(X, y)
print("Intercept",model.intercept_)
print("X1 coefficient",model.coef_[0][0])
print("X2 coefficient",model.coef_[0][1])



matplotlib.pyplot.figure()

positive=X[y==1]
negative=X[y==-1]

matplotlib.pyplot.scatter(positive[:, 0], positive[:, 1],color="blue",marker="+")
matplotlib.pyplot.scatter(negative[:, 0], negative[:, 1],color="lightgreen", marker="+")


matplotlib.pyplot.xlabel("x_1")
matplotlib.pyplot.ylabel("x_2")
matplotlib.pyplot.legend(["positive","negative"])
matplotlib.pyplot.show()


