import matplotlib.pyplot

import numpy as np
import pandas as pd
df = pd.read_csv("week2.csv")
print(df.head())
X1=df.iloc[:,0]
X2=df.iloc[:,1]
X=np.column_stack((X1,X2))
y=df.iloc[:,2]

matplotlib.pyplot.figure()

positive=X[y==1]
negative=X[y==-1]

QUESTION1=False
QUESTION2=False

if(QUESTION1):
    matplotlib.pyplot.scatter(positive[:, 0], positive[:, 1],color="blue",marker="+")
    matplotlib.pyplot.scatter(negative[:, 0], negative[:, 1],color="lightgreen", marker="+")

    matplotlib.pyplot.xlabel("x_1")
    matplotlib.pyplot.ylabel("x_2")
    matplotlib.pyplot.legend(["positive","negative"])
    matplotlib.pyplot.show()

if(QUESTION2):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression().fit(X, y)
    print("Intercept",model.intercept_)
    print("X1 coefficient",model.coef_[0][0])
    print("X2 coefficient",model.coef_[0][1])

    intercept = -model.intercept_[0] / model.coef_[0][1]
    slope = -model.coef_[0][0] / model.coef_[0][1]

    # y = mx + c
    # y = (slope*x) + intercept
    decision_boundary = (slope*X[:, 0]) + intercept
    matplotlib.pyplot.plot(X[:, 0], decision_boundary, color="red", linewidth=2, label="Decision Boundary")

    y_prediction=model.predict(X)
    correctPositive=X[(y==1) & (y_prediction==1)]
    wrongPosive=X[(y==1) & (y_prediction==-1)]
    correctNegative=X[(y==-1) & (y_prediction==-1)]
    wrongNegative=X[(y==-1) & (y_prediction==1)]

    matplotlib.pyplot.scatter(correctPositive[:, 0], correctPositive[:, 1], color="blue", marker="+", label="Correct +1 Prediction")
    matplotlib.pyplot.scatter(wrongPosive[:, 0], wrongPosive[:, 1], color="darkblue", marker="+", label="Incorrect +1 Prediction")
    matplotlib.pyplot.scatter(correctNegative[:, 0], correctNegative[:, 1], color="lightgreen", marker="+", label="Correct -1 Prediction")
    matplotlib.pyplot.scatter(wrongNegative[:, 0], wrongNegative[:, 1], color="green", marker="+", label="Incorrect -1 Prediction")
    matplotlib.pyplot.legend(["decision boundary","correct +1 prediction","incorrect +1 prediction","correct -1 prediction","incorrect -1 prediction"])


from sklearn.svm import LinearSVC
axes = matplotlib.pyplot.subplots(2, 3, figsize=(15, 10))[1]
axes = axes.ravel()
C_values=[0.00001,0.001,0.01, 1.0, 100.0]
for i in range(0,len(C_values)):
    C_val = C_values[i]
    # Train SVM model
    model_svm = LinearSVC(C=C_val).fit(X, y)
    
    # Make predictions
    y_pred = model_svm.predict(X)
    
    # Calculate decision boundary
    intercept_svm = -model_svm.intercept_[0] / model_svm.coef_[0][1]
    slope_svm = -model_svm.coef_[0][0] / model_svm.coef_[0][1]
    decision_boundary_svm = (slope_svm * X[:, 0]) + intercept_svm
    
    # Plot on subplot
    ax = axes[i]
    
    # Plot data points with predictions
    correctPositive = X[(y == 1) & (y_pred == 1)]
    wrongPosive = X[(y == 1) & (y_pred == -1)]
    correctNegative = X[(y == -1) & (y_pred == -1)]
    wrongNegative = X[(y == -1) & (y_pred == 1)]
    
    ax.scatter(correctPositive[:, 0], correctPositive[:, 1], color="blue", marker="+", label="Correct +1 Prediction")
    ax.scatter(wrongPosive[:, 0], wrongPosive[:, 1], color="darkblue", marker="+", label="Incorrect +1 Prediction")
    ax.scatter(correctNegative[:, 0], correctNegative[:, 1], color="lightgreen", marker="+", label="Correct -1 Prediction")
    ax.scatter(wrongNegative[:, 0], wrongNegative[:, 1], color="green", marker="+", label="Incorrect -1 Prediction")
    
    # Plot decision boundary
    ax.plot(X[:, 0], decision_boundary_svm, color="red", linewidth=2, label="Decision Boundary")
    
    ax.set_title("SVM with C="+str(C_val))
    ax.set_xlabel("x_1")
    ax.set_ylabel("x_2")
    ax.legend(["decision boundary","correct +1 prediction","incorrect +1 prediction","correct -1 prediction","incorrect -1 prediction"])

# Remove empty subplot if needed
axes[5].set_visible(False)

matplotlib.pyplot.tight_layout()
matplotlib.pyplot.show()