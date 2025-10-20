import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D

import numpy
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
import sklearn.preprocessing
import sklearn.linear_model
df = pandas.read_csv("week3.csv")

CLIPPED=True

QUESTIONIA=False
QUESTIONIB=False
QUESTIONIC=False
QUESTIONIE=False
QUESTIONIIA=False
QUESTIONIIC=True

C_values = [0.0001,0.001,0.1,1,10,100,1000]
X1=df.iloc[:,0]
X2=df.iloc[:,1]
X=numpy.column_stack((X1,X2))
y=df.iloc[:,2]
polynomialFeatures = sklearn.preprocessing.PolynomialFeatures(degree=5)
X_poly = polynomialFeatures.fit_transform(X)
features = polynomialFeatures.get_feature_names_out(["X1","X2"])
print("Features:",features)

if(QUESTIONIA):
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(111,projection="3d")
    ax.scatter(X[:,0] ,X[:,1],y)

    matplotlib.pyplot.xlabel("X1")
    matplotlib.pyplot.ylabel("X2")
    ax.set_zlabel("y")

    matplotlib.pyplot.show()

if(QUESTIONIB or QUESTIONIC or QUESTIONIE):
    models={}
    


    for C in C_values:
        print("C=",C)
        alpha=1/(2*C)
        print("alpha=",alpha)
        lasso = sklearn.linear_model.Lasso(alpha)
        lasso.fit(X_poly, y)
        models[C]=(lasso)
        print("Intercept:",lasso.intercept_)

        for f,c in zip(features,lasso.coef_):
            print(f,":",c)

        print("")
    
        
if(QUESTIONIC or QUESTIONIE):
    Xtest=[]
    grid = numpy.linspace(-5,5)
    for i  in grid:
        for j in grid:
            Xtest.append([i,j])
    Xtest=numpy.array(Xtest)

    for C in models:
        model = models[C]
        Xtest_poly = polynomialFeatures.transform(Xtest)
        y_prediction = model.predict(Xtest_poly)
        
        
        
if(QUESTIONIC):

    fig = matplotlib.pyplot.figure(figsize=(15,12))
    
    # Create subplots in a grid - 2 rows, 3 columns for 6 models
    plot_num = 1
    for C in models:
        model = models[C]
        Xtest_poly = polynomialFeatures.transform(Xtest)
        y_prediction = model.predict(Xtest_poly)
        
        # Create subplot at the correct position
        ax = fig.add_subplot(2, 4, plot_num, projection="3d")
        
        X1_grid = Xtest[:, 0].reshape(len(grid), len(grid))
        X2_grid = Xtest[:, 1].reshape(len(grid), len(grid))
        y_prediction_grid = y_prediction.reshape(len(grid), len(grid))
        if(CLIPPED):
            y_prediction_clipped = numpy.clip(y_prediction_grid, -2, 2)
            surface = ax.plot_surface(X1_grid, X2_grid, y_prediction_clipped, alpha=0.7, cmap="viridis")
        else:
            surface = ax.plot_surface(X1_grid, X2_grid, y_prediction_grid, alpha=0.7, cmap="viridis")
        
        # Plot the training data as scatter points
        scatter = ax.scatter(X[:, 0], X[:, 1], y, color="red", marker="o", s=30, label="Training Data")
        
        # Set labels and title
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_zlabel("y")
        title="C="+str(C)+" (α="+str(round(1/(2*C), 4))+")"
        ax.set_title(title)
        
        # Set limits for the axes
        if(CLIPPED):
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_zlim(-1, 1)
        
        ax.legend()
        ax.view_init(elev=20, azim=45)
        
        print("Model with C=",C ,"(α=",1/(2*C),")")
        nonZeroCoeffs = 0
        for coef in model.coef_:
            if coef != 0:
                nonZeroCoeffs += 1
        print("Number of non-zero coefficients: ",nonZeroCoeffs)
        print("Model intercept: ",model.intercept_)
        
        non_zero_coefs = [(f, c) for f, c in zip(features, model.coef_) if c != 0]
        if non_zero_coefs:
            print("Non-zero coefficients:")
            for feature, coef in non_zero_coefs:
                print("  ", feature, ": ", round(coef, 4))
        else:
            print("All coefficients are zero")
        print("")
        
        plot_num += 1  # Move to next subplot position
    
    fig.tight_layout()
    matplotlib.pyplot.show()

if(QUESTIONIE):
    print("\n" + "="*60)
    print("QUESTION E - RIDGE REGRESSION COMPARISON")
    print("="*60)
    
    # Train Ridge Regression models
    ridge_models = {}
    
    for C in C_values:
        print("Ridge Regression - C=",C)
        alpha=1/(2*C)
        print("alpha=",alpha)
        ridge = sklearn.linear_model.Ridge(alpha)
        ridge.fit(X_poly, y)
        ridge_models[C]=(ridge)
        print("Intercept:",ridge.intercept_)

        for f,c in zip(features,ridge.coef_):
            print(f,":",c)

        print("")
    
    # Create comparison plots for Lasso vs Ridge
    fig = matplotlib.pyplot.figure(figsize=(20,12))
    
    plot_num = 1
    for C in C_values:
        # Lasso model
        lasso_model = models[C]
        ridge_model = ridge_models[C]
        
        # Generate predictions for both models
        Xtest_poly = polynomialFeatures.transform(Xtest)
        lasso_prediction = lasso_model.predict(Xtest_poly)
        ridge_prediction = ridge_model.predict(Xtest_poly)
        
        # Lasso subplot
        ax1 = fig.add_subplot(2, len(C_values), plot_num, projection="3d")
        lasso_pred_grid = lasso_prediction.reshape(len(grid), len(grid))
        X1_grid = Xtest[:, 0].reshape(len(grid), len(grid))
        X2_grid = Xtest[:, 1].reshape(len(grid), len(grid))
        if(CLIPPED):
            lasso_pred_clipped = numpy.clip(lasso_pred_grid, -2, 2)
            surface1 = ax1.plot_surface(X1_grid, X2_grid, lasso_pred_clipped, alpha=0.7, cmap="viridis")
        else:
            surface1 = ax1.plot_surface(X1_grid, X2_grid, lasso_pred_grid, alpha=0.7, cmap="viridis")
        scatter1 = ax1.scatter(X[:, 0], X[:, 1], y, color="red", marker="o", s=20, label="Training Data")
        ax1.set_xlabel("X1")
        ax1.set_ylabel("X2")
        ax1.set_zlabel("y")
        ax1.set_title("Lasso C="+str(C)+" (α="+str(round(1/(2*C), 4))+")")
        ax1.legend()
        ax1.view_init(elev=20, azim=45)
        
        # Ridge subplot
        ax2 = fig.add_subplot(2, len(C_values), plot_num + len(C_values), projection="3d")
        ridge_pred_grid = ridge_prediction.reshape(len(grid), len(grid))
        if(CLIPPED):
            ridge_pred_clipped = numpy.clip(ridge_pred_grid, -2, 2)
            surface2 = ax2.plot_surface(X1_grid, X2_grid, ridge_pred_clipped, alpha=0.7, cmap="plasma")
        else:
            surface2 = ax2.plot_surface(X1_grid, X2_grid, ridge_pred_grid, alpha=0.7, cmap="plasma")
        scatter2 = ax2.scatter(X[:, 0], X[:, 1], y, color="red", marker="o", s=20, label="Training Data")
        ax2.set_xlabel("X1")
        ax2.set_ylabel("X2")
        ax2.set_zlabel("y")
        ax2.set_title("Ridge C="+str(C)+" (α="+str(round(1/(2*C), 4))+")")
        ax2.legend()
        ax2.view_init(elev=20, azim=45)
        
        # Print comparison statistics
        print("\nCOMPARISON for C=", C)
        print("Lasso - Non-zero coefficients: ", numpy.sum(lasso_model.coef_ != 0))
        print("Ridge - Non-zero coefficients: ", numpy.sum(ridge_model.coef_ != 0))
        print("Lasso - Sum of abs coefficients: ", round(numpy.sum(numpy.abs(lasso_model.coef_)), 4))
        print("Ridge - Sum of abs coefficients: ", round(numpy.sum(numpy.abs(ridge_model.coef_)), 4))
        print("Lasso - Max coefficient: ", round(numpy.max(numpy.abs(lasso_model.coef_)), 4))
        print("Ridge - Max coefficient: ", round(numpy.max(numpy.abs(ridge_model.coef_)), 4))
        
        plot_num += 1
    
    fig.tight_layout()
    matplotlib.pyplot.show()
    
    # Print summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON: LASSO vs RIDGE")
    print("="*60)
    for C in C_values:
        lasso_model = models[C]
        ridge_model = ridge_models[C]
        lasso_nonzero = numpy.sum(lasso_model.coef_ != 0)
        ridge_nonzero = numpy.sum(ridge_model.coef_ != 0)
        print("C=", C, ": Lasso has", lasso_nonzero, "non-zero coeffs, Ridge has", ridge_nonzero, "non-zero coeffs")


if(QUESTIONIIA):
    mean_scores = []
    std_scores = []
    C_values=[0.001,0.002,0.005, 0.01,0.02,0.05, 0.1,0.2, 0.5, 1, 2,3,4, 5, 10, 20, 50, 100, 200, 500, 1000,2000,5000,10000,20000,50000,100000]
    for C in C_values:
        alpha = 1/(2*C)
        lasso = sklearn.linear_model.Lasso(alpha)
        
        # Use negative MSE as score (so higher is better)
        # cross_val_score returns negative MSE for scoring='neg_mean_squared_error'
        scores = cross_val_score(lasso, X_poly, y, cv=5, scoring='neg_mean_squared_error')
        
        # Convert to positive MSE for plotting
        mse_scores = -scores
        mean_mse = numpy.mean(mse_scores)
        std_mse = numpy.std(mse_scores)
        
        mean_scores.append(mean_mse)
        std_scores.append(std_mse)
        
        print("C=", C, " - Mean MSE:", round(mean_mse, 4), "+/-", round(std_mse, 4))
    
    # Create the errorbar plot
    fig, ax = matplotlib.pyplot.subplots(figsize=(12, 8))
    
    # Convert to numpy arrays for easier handling
    C_array = numpy.array(C_values)
    mean_array = numpy.array(mean_scores)
    std_array = numpy.array(std_scores)
    
    # Plot with error bars
    ax.errorbar(C_array, mean_array, yerr=std_array, fmt='o-', linewidth=2, 
                markersize=8, capsize=5, capthick=2, elinewidth=2, 
                label='5-fold CV MSE ± 1 std dev')
    
    # Set log scale for x-axis since C values span multiple orders of magnitude
    ax.set_xscale('log')
    

    
    # Labels and formatting
    ax.set_xlabel('Regularization Parameter C', fontsize=12)
    ax.set_ylabel('Mean Squared Error (MSE)', fontsize=12)
    ax.set_title('Lasso Regression: 5-Fold Cross-Validation Error vs C', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    
    fig.tight_layout()
    matplotlib.pyplot.show()
    
if(QUESTIONIIC):
    mean_scores = []
    std_scores = []
    C_values=[0.001,0.002,0.005, 0.01,0.02,0.05, 0.1,0.2, 0.5, 1, 2,3,4, 5, 10, 20, 50, 100, 200, 500, 1000,2000,5000,10000,20000,50000,100000]
    for C in C_values:
        alpha = 1/(2*C)
        ridge = sklearn.linear_model.Ridge(alpha)
        
        # Use negative MSE as score (so higher is better)
        # cross_val_score returns negative MSE for scoring='neg_mean_squared_error'
        scores = cross_val_score(ridge, X_poly, y, cv=5, scoring='neg_mean_squared_error')
        
        # Convert to positive MSE for plotting
        mse_scores = -scores
        mean_mse = numpy.mean(mse_scores)
        std_mse = numpy.std(mse_scores)
        
        mean_scores.append(mean_mse)
        std_scores.append(std_mse)
        
        print("C=", C, " - Mean MSE:", round(mean_mse, 4), "+/-", round(std_mse, 4))
    
    # Create the errorbar plot
    fig, ax = matplotlib.pyplot.subplots(figsize=(12, 8))
    
    # Convert to numpy arrays for easier handling
    C_array = numpy.array(C_values)
    mean_array = numpy.array(mean_scores)
    std_array = numpy.array(std_scores)
    
    # Plot with error bars
    ax.errorbar(C_array, mean_array, yerr=std_array, fmt='o-', linewidth=2, 
                markersize=8, capsize=5, capthick=2, elinewidth=2, 
                label='5-fold CV MSE ± 1 std dev')
    
    # Set log scale for x-axis since C values span multiple orders of magnitude
    ax.set_xscale('log')
    

    
    # Labels and formatting
    ax.set_xlabel('Regularization Parameter C', fontsize=12)
    ax.set_ylabel('Mean Squared Error (MSE)', fontsize=12)
    ax.set_title('ridge Regression: 5-Fold Cross-Validation Error vs C', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    
    fig.tight_layout()
    matplotlib.pyplot.show()
    
    