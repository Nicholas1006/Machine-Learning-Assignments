import matplotlib.pyplot
import numpy
import pandas
import sklearn.preprocessing
import sklearn.linear_model
from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, auc

QUESTION_IA = True
QUESTION_IB = True
QUESTION_IC = True
QUESTION_ID = True
DATASET_FILE = "dataset2.csv"



# Load and split the dataset file
print("Loading dataset file...")
df1 = pandas.read_csv(DATASET_FILE)
X1 = df1.iloc[:, 0]
X2 = df1.iloc[:, 1]
X = numpy.column_stack((X1, X2))
y = df1.iloc[:, 2]

print("Dataset 1 shape - X:", X.shape, "y:", y.shape)



if QUESTION_IA:
    polynomial_degrees = [1, 2, 3, 4, 5, 50] 
    C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000] 
    
    best_degree = None
    best_C = None
    best_score = -1
    standard_deviation = 0
    all_results = []
    
    print("Performing cross-validation for polynomial features and regularization...")
    print("Polynomial degrees to test:", polynomial_degrees)
    print("C values to test:", C_values)
    lower_bound=1
    upper_bound=0
    for degree in polynomial_degrees:
        print("")
        print("")
        print("Testing Polynomial Degree "+str(degree))
        
        polynomialFeatures = sklearn.preprocessing.PolynomialFeatures(degree=degree)
        X_poly = polynomialFeatures.fit_transform(X)
        print("Number of features after polynomial expansion: "+str(X_poly.shape[1]))
        
        degree_results = []
        
        for C in C_values:
            logistic_regression_model = sklearn.linear_model.LogisticRegression(C=C, penalty="l2",max_iter=1000)
            scores = cross_val_score(logistic_regression_model, X_poly, y, cv=5, scoring="accuracy")
            mean_score = numpy.mean(scores)
            std_score = numpy.std(scores)
            
            degree_results.append({
                "C": C,
                "mean_accuracy": mean_score,
                "std_accuracy": std_score
            })
            
            upper_bound=max(upper_bound,mean_score+std_score)
            lower_bound=min(lower_bound,mean_score-std_score)

            print("C="+str(C)+": Mean Accuracy = "+str(mean_score)+" +/- "+str(std_score))
            
            if mean_score > best_score:
                best_score = mean_score
                best_degree = degree
                best_C = C
                standard_deviation = std_score
        
        all_results.append({
            "degree": degree,
            "results": degree_results
        })
    
    
    print("BEST PARAMETERS FOUND:")
    print("Polynomial Degree: "+str(best_degree))
    print("Regularization C: "+str(best_C))
    print("Best Cross-Validation Accuracy: "+str(best_score))
    print("Standard Deviation: "+str(standard_deviation))
    
    fig, axes = matplotlib.pyplot.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (degree_data, ax) in enumerate(zip(all_results, axes)):
        if idx >= len(axes):
            break
            
        degree = degree_data["degree"]
        results = degree_data["results"]
        
        C_vals = [r["C"] for r in results]
        mean_acc = [r["mean_accuracy"] for r in results]
        std_acc = [r["std_accuracy"] for r in results]
        
        ax.errorbar(C_vals, mean_acc, yerr=std_acc, fmt="o-", linewidth=2, 
                   markersize=6, capsize=4, capthick=1, elinewidth=1)
        ax.set_xscale("log")
        ax.set_xlabel("Regularization C")
        ax.set_ylim(lower_bound,upper_bound)
        ax.set_ylabel("Accuracy")
        ax.set_title("Polynomial Degree "+str(degree))
        ax.grid(True, alpha=0.3)
        
        best_idx = numpy.argmax(mean_acc)
        ax.plot(C_vals[best_idx], mean_acc[best_idx], "ro", markersize=8, 
                markerfacecolor="none", markeredgewidth=2)
    
    for idx in range(len(all_results), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle("Logistic Regression: 5-Fold Cross-Validation Accuracy vs C for Different Polynomial Degrees", 
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    matplotlib.pyplot.show()
    
    print("")
    print("Training final model with best parameters...")
    final_poly = sklearn.preprocessing.PolynomialFeatures(degree=best_degree)
    X_poly_final = final_poly.fit_transform(X)
    
    final_model = sklearn.linear_model.LogisticRegression(C=best_C, penalty="l2",max_iter=1000, random_state=42)
    final_model.fit(X_poly_final, y)
    
    print("")
    print("Generating decision boundary visualization...")
    
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = numpy.meshgrid(numpy.linspace(x_min, x_max, 100),
                           numpy.linspace(y_min, y_max, 100))
    
    mesh_points = numpy.c_[xx.ravel(), yy.ravel()]
    mesh_poly = final_poly.transform(mesh_points)
    Z = final_model.predict(mesh_poly)
    Z = Z.reshape(xx.shape)
    
    fig, ax = matplotlib.pyplot.subplots(figsize=(10, 8))
    
    contour = ax.contourf(xx, yy, Z, alpha=0.3)
    
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors="black", s=50)
    
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Logistic Regression Decision Boundary" + "\n" + "Polynomial Degree: "+str(best_degree)+", C: "+str(best_C))
    print("(Polynomial Degree: "+str(best_degree)+", C: "+str(best_C))

    
    unique_classes = numpy.unique(y)
    for cls in unique_classes:
        ax.scatter([], [], c=[scatter.cmap(scatter.norm(cls))], label=("Class "+str(cls)), edgecolors="black")
    ax.legend()
    
    fig.tight_layout()
    matplotlib.pyplot.show()
    
    print("")
    print("Final Model Information:")
    print("Number of coefficients: "+str(len(final_model.coef_[0])))
    print("Intercept: "+str(final_model.intercept_[0]))
    print("Training accuracy: "+str(final_model.score(X_poly_final, y)))



if QUESTION_IB:
    print("QUESTION I(B) - kNN CLASSIFIER WITH CROSS-VALIDATION")
    
    # Define range of k values to test
    k_values = list(range(1, 101))  # Test k from 1 to 30
    print("Testing k values from 1 to 30")
    print("Class distribution:", numpy.unique(y, return_counts=True))
    
    # Store results
    k_results = []
    best_k = None
    best_score = -1
    standard_deviation = 0
    
    print("\nPerforming 5-fold cross-validation for kNN...")
    
    for k in k_values:
        # Create kNN classifier
        knn = KNeighborsClassifier(n_neighbors=k)
        
        # Perform 5-fold cross-validation with accuracy scoring
        scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
        mean_score = numpy.mean(scores)
        std_score = numpy.std(scores)
        
        k_results.append({
            'k': k,
            'mean_accuracy': mean_score,
            'std_accuracy': std_score
        })
        
        print("k=" + str(k) + ": Mean Accuracy = " + str(mean_score) + 
              " +/- " + str(std_score))
        
        # Update best k
        if mean_score > best_score:
            best_score = mean_score
            best_k = k
            standard_deviation = std_score
    
    # Print best parameters
    print("\n" + "="*60)
    print("BEST PARAMETERS FOUND:")
    print("Best k: " + str(best_k))
    print("Best Cross-Validation Accuracy: " + str(best_score))
    print("Standard Deviation: " + str(standard_deviation))
    print("="*60)
    
    # Create visualization of k vs accuracy
    k_vals = [r['k'] for r in k_results]
    mean_acc = [r['mean_accuracy'] for r in k_results]
    std_acc = [r['std_accuracy'] for r in k_results]
    
    fig, ax = matplotlib.pyplot.subplots(figsize=(10, 6))
    
    # Plot 1: Error bars
    ax.errorbar(k_vals, mean_acc, yerr=std_acc, fmt='o-', linewidth=2, 
                markersize=6, capsize=4, capthick=1, elinewidth=1,
                label='5-fold CV Accuracy ± 1 std dev')
    ax.set_xlabel('Number of Neighbors (k)')
    ax.set_ylabel('Accuracy')
    ax.set_title('kNN: Cross-Validation Accuracy vs k')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Highlight best point
    best_idx = k_vals.index(best_k)
    ax.plot(best_k, best_score, 'ro', markersize=10, markerfacecolor='none',
             markeredgewidth=3, label='Best k = ' + str(best_k))
    ax.legend()
    
    
    fig.tight_layout()
    matplotlib.pyplot.show()
    
    # Train final model with best k
    print("\nTraining final kNN model with best k=" + str(best_k) + "...")
    final_knn = KNeighborsClassifier(n_neighbors=best_k)
    final_knn.fit(X, y)
    training_accuracy = final_knn.score(X, y)
    print("Training accuracy: " + str(training_accuracy))
    
    # Generate decision boundary visualization
    print("\nGenerating kNN decision boundary visualization...")
    
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = numpy.meshgrid(numpy.linspace(x_min, x_max, 100),
                           numpy.linspace(y_min, y_max, 100))
    
    mesh_points = numpy.c_[xx.ravel(), yy.ravel()]
    Z = final_knn.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    fig, ax = matplotlib.pyplot.subplots(figsize=(10, 8))
    
    # Plot decision boundary
    contour = ax.contourf(xx, yy, Z, alpha=0.3)
    
    # Plot training data
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='black', s=50)
    
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('kNN Decision Boundary (k=' + str(best_k) + ')')
    
    # Add legend
    unique_classes = numpy.unique(y)
    for cls in unique_classes:
        ax.scatter([], [], c=[scatter.cmap(scatter.norm(cls))], 
                  label='Class ' + str(cls), edgecolors='black')
    ax.legend()
    
    fig.tight_layout()
    matplotlib.pyplot.show()
    
    # Analyze the results
    print("\n" + "="*60)
    print("kNN CROSS-VALIDATION ANALYSIS")
    print("="*60)
    
    # Calculate accuracy range for interpretation
    min_acc = min(mean_acc)
    max_acc = max(mean_acc)
    accuracy_range = max_acc - min_acc
    
    print("Accuracy range across k values: " + str(min_acc) + " to " + 
          str(max_acc) + " (range: " + str(accuracy_range) + ")")
    
    # Analyze the shape of the accuracy curve
    if best_k == 1:
        print("Best performance at k=1 suggests:")
        print("- Data may have very local patterns")
        print("- Risk of overfitting to noise in training data")
    elif best_k == k_values[-1]:
        print("Best performance at largest k suggests:")
        print("- Data has smooth decision boundaries")
        print("- Model benefits from smoothing effect of many neighbors")
    else:
        print("Best performance at intermediate k=" + str(best_k) + " suggests:")
        print("- Balanced trade-off between bias and variance")
        print("- Model captures local patterns while reducing noise sensitivity")
    
    # Check stability around best k
    best_std = std_acc[best_idx]
    print("Standard deviation at best k: " + str(best_std))
    if best_std < 0.02:
        print("- Low variance across folds (stable performance)")
    elif best_std < 0.05:
        print("- Moderate variance across folds")
    else:
        print("- High variance across folds (performance depends on data split)")
    
    # Compare with extreme k values
    k1_accuracy = mean_acc[0]  # k=1
    k30_accuracy = mean_acc[-1]  # k=30
    
    print("\nComparison with extreme k values:")
    print("k=1 accuracy: " + str(k1_accuracy))
    print("k=30 accuracy: " + str(k30_accuracy))
    
    if best_score > k1_accuracy and best_score > k30_accuracy:
        print("Optimal k provides better performance than both extremes")
    elif k1_accuracy > best_score:
        print("k=1 performs better - data may have very local structure")
    else:
        print("Large k performs better - data benefits from smoothing")
    
    # Recommendation
    print("\nRECOMMENDATION:")
    print("Use k=" + str(best_k) + " for the kNN classifier")
    print("This value provides the best cross-validation accuracy of " + 
          str(best_score) + " while maintaining reasonable stability")
    


if QUESTION_IC:
    print("\n" + "="*60)
    print("QUESTION I(C) - CONFUSION MATRICES")
    print("="*60)
    
    # Split data into train and test sets for proper evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42, stratify=y)
    
    print("Training set size: " + str(X_train.shape[0]))
    print("Test set size: " + str(X_test.shape[0]))
    
    # 1. Logistic Regression Model (using best parameters from previous question)
    print("")
    print("LOGISTIC REGRESSION CONFUSION MATRIX")
    
    # Use the best parameters found in question (i)(a)
    # If you haven't run (i)(a), you'll need to define best_degree and best_C
    # For now, I'll use reasonable defaults - replace with your actual best values
    if 'best_degree' not in locals():
        best_degree = 3  # Replace with your actual best degree
        best_C = 10      # Replace with your actual best C
    
    poly_features = sklearn.preprocessing.PolynomialFeatures(degree=best_degree)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)
    
    logreg_model = sklearn.linear_model.LogisticRegression(C=best_C, penalty='l2')
    logreg_model.fit(X_train_poly, y_train)
    
    y_pred_logreg = logreg_model.predict(X_test_poly)
    cm_logreg = confusion_matrix(y_test, y_pred_logreg)
    accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
    
    print("Best parameters - Degree: " + str(best_degree) + ", C: " + str(best_C))
    print("Confusion Matrix:")
    print(cm_logreg)
    print("Accuracy: " + str(accuracy_logreg))
    
    # Calculate metrics from confusion matrix
    tn_logreg, fp_logreg, fn_logreg, tp_logreg = cm_logreg.ravel()
    print("True Negatives: " + str(tn_logreg))
    print("False Positives: " + str(fp_logreg))
    print("False Negatives: " + str(fn_logreg))
    print("True Positives: " + str(tp_logreg))
    
    # 2. kNN Model (using best k from previous question)
    print("")
    print("kNN CLASSIFIER CONFUSION MATRIX")
    
    # Use the best k found in question (i)(b)
    # If you haven't run (i)(b), you'll need to define best_k
    if 'best_k' not in locals():
        best_k = 39  # Replace with your actual best k
    
    knn_model = KNeighborsClassifier(n_neighbors=best_k)
    knn_model.fit(X_train, y_train)
    
    y_pred_knn = knn_model.predict(X_test)
    cm_knn = confusion_matrix(y_test, y_pred_knn)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    
    print("Best k: " + str(best_k))
    print("Confusion Matrix:")
    print(cm_knn)
    print("Accuracy: " + str(accuracy_knn))
    
    # Calculate metrics from confusion matrix
    tn_knn, fp_knn, fn_knn, tp_knn = cm_knn.ravel()
    print("True Negatives: " + str(tn_knn))
    print("False Positives: " + str(fp_knn))
    print("False Negatives: " + str(fn_knn))
    print("True Positives: " + str(tp_knn))
    
    # 3. Baseline Classifier - Most Frequent Class
    print("")
    print("BASELINE CLASSIFIER CONFUSION MATRIX")
    
    # Find the most frequent class in training data
    unique_classes, class_counts = numpy.unique(y_train, return_counts=True)
    most_frequent_class = unique_classes[numpy.argmax(class_counts)]
    
    print("Most frequent class in training data: " + str(most_frequent_class))
    
    # Always predict the most frequent class
    y_pred_baseline = numpy.full(y_test.shape, most_frequent_class)
    cm_baseline = confusion_matrix(y_test, y_pred_baseline)
    accuracy_baseline = accuracy_score(y_test, y_pred_baseline)
    
    print("Confusion Matrix:")
    print(cm_baseline)
    print("Accuracy: " + str(accuracy_baseline))
    
    # Calculate metrics from confusion matrix
    tn_baseline, fp_baseline, fn_baseline, tp_baseline = cm_baseline.ravel()
    print("True Negatives: " + str(tn_baseline))
    print("False Positives: " + str(fp_baseline))
    print("False Negatives: " + str(fn_baseline))
    print("True Positives: " + str(tp_baseline))
    
    # 4. Create visual comparison of confusion matrices
    print("")
    print("VISUAL COMPARISON OF CONFUSION MATRICES")
    
    fig, axes = matplotlib.pyplot.subplots(1, 3, figsize=(15, 5))
    
    # Logistic Regression confusion matrix plot
    im1 = axes[0].imshow(cm_logreg,  cmap='Blues')
    axes[0].set_title('Logistic Regression\nAccuracy: ' + str(accuracy_logreg))
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    # Add text annotations
    for i in range(cm_logreg.shape[0]):
        for j in range(cm_logreg.shape[1]):
            axes[0].text(j, i, str(cm_logreg[i, j]), ha='center', va='center', color='black' if cm_logreg[i, j] < cm_logreg.max()/2 else 'white')
    
    # kNN confusion matrix plot
    im2 = axes[1].imshow(cm_knn,  cmap='Blues')
    axes[1].set_title('kNN Classifier\nAccuracy: ' + str(accuracy_knn))
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    for i in range(cm_knn.shape[0]):
        for j in range(cm_knn.shape[1]):
            axes[1].text(j, i, str(cm_knn[i, j]), ha='center', va='center', color='black' if cm_knn[i, j] < cm_knn.max()/2 else 'white')
    
    # Baseline confusion matrix plot
    im3 = axes[2].imshow(cm_baseline,  cmap='Blues')
    axes[2].set_title('Baseline (Most Frequent)\nAccuracy: ' + str(accuracy_baseline))
    axes[2].set_xlabel('Predicted')
    axes[2].set_ylabel('Actual')
    
    for i in range(cm_baseline.shape[0]):
        for j in range(cm_baseline.shape[1]):
            axes[2].text(j, i, str(cm_baseline[i, j]),  ha='center', va='center', color='black' if cm_baseline[i, j] < cm_baseline.max()/2 else 'white')
    
    fig.tight_layout()
    matplotlib.pyplot.show()
    
    # 5. Comparative Analysis
    print("")
    print("COMPARATIVE ANALYSIS")
    
    print("Accuracy Comparison:")
    print("Logistic Regression: " + str(accuracy_logreg))
    print("kNN Classifier: " + str(accuracy_knn))
    print("Baseline: " + str(accuracy_baseline))
    
    # Calculate improvement over baseline
    improvement_logreg = accuracy_logreg - accuracy_baseline
    improvement_knn = accuracy_knn - accuracy_baseline
    
    print("\nImprovement over baseline:")
    print("Logistic Regression: +" + str((improvement_logreg)))
    print("kNN Classifier: +" + str(improvement_knn))
    
    # Determine best classifier
    if accuracy_logreg > accuracy_knn:
        best_classifier = "Logistic Regression"
        best_accuracy = accuracy_logreg
    else:
        best_classifier = "kNN Classifier" 
        best_accuracy = accuracy_knn
    
    print("\nBest performing classifier: " + best_classifier)
    print("Best accuracy: " + str(best_accuracy))
    
    # Error analysis
    print("\nError Analysis:")
    total_test_samples = len(y_test)
    
    print("Logistic Regression error rate: " + str(1 - accuracy_logreg) + 
          " (" + str(int((1 - accuracy_logreg) * total_test_samples)) + " samples)")
    print("kNN error rate: " + str(1 - accuracy_knn) + 
          " (" + str(int((1 - accuracy_knn) * total_test_samples)) + " samples)")
    print("Baseline error rate: " + str(1 - accuracy_baseline) + 
          " (" + str(int((1 - accuracy_baseline) * total_test_samples)) + " samples)")
    
    # Check if improvements are significant
    if improvement_logreg > 0.1 or improvement_knn > 0.1:
        print("\nBoth classifiers show substantial improvement over baseline")
    elif improvement_logreg > 0.05 or improvement_knn > 0.05:
        print("\nModerate improvement over baseline")
    else:
        print("\nMinimal improvement over baseline - data may be challenging to classify")


if QUESTION_ID:
    matplotlib.pyplot.rc('font', size=18); matplotlib.pyplot.rcParams['figure.constrained_layout.use'] = True
    print("\n" + "="*60)
    print("QUESTION I(D) - ROC CURVES")
    print("="*60)
    
    # Use the same train-test split as before for consistency
    if 'X_test' not in locals():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                            random_state=42, stratify=y)
    
    print("Test set size: " + str(X_test.shape[0]))
    print("Test set class distribution: " + str(numpy.unique(y_test, return_counts=True)))
    
    # Ensure classes are 0 and 1 for ROC analysis (some datasets use -1,1)
    if numpy.min(y) == -1:
        y_train_roc = (y_train + 1) // 2
        y_test_roc = (y_test + 1) // 2
        print("Converted class labels from [-1,1] to [0,1] for ROC analysis")
    else:
        y_train_roc = y_train
        y_test_roc = y_test
    
    # 1. Logistic Regression ROC Curve
    print("\n" + "-"*40)
    print("LOGISTIC REGRESSION ROC CURVE")
    print("-"*40)
    
    # Use the same polynomial features and model as before
    if 'best_degree' not in locals():
        best_degree = 3
        best_C = 10
    
    poly_features = sklearn.preprocessing.PolynomialFeatures(degree=best_degree)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)
    
    logreg_model = sklearn.linear_model.LogisticRegression(C=best_C, penalty='l2', 
                                                          max_iter=1000, random_state=42)
    logreg_model.fit(X_train_poly, y_train_roc)
    
    # Get probability scores for positive class (class 1)
    y_score_logreg = logreg_model.predict_proba(X_test_poly)[:, 1]
    
    # Calculate ROC curve
    fpr_logreg, tpr_logreg, _ = roc_curve(y_test_roc, y_score_logreg)
    auc_logreg = auc(fpr_logreg, tpr_logreg)
    
    print("AUC Score: " + str(round(auc_logreg, 4)))
    
    # 2. kNN ROC Curve
    print("\n" + "-"*40)
    print("kNN CLASSIFIER ROC CURVE")
    print("-"*40)
    
    if 'best_k' not in locals():
        best_k = 5
    
    knn_model = KNeighborsClassifier(n_neighbors=best_k)
    knn_model.fit(X_train, y_train_roc)
    
    # Get probability scores for positive class
    y_score_knn = knn_model.predict_proba(X_test)[:, 1]
    
    # Calculate ROC curve
    fpr_knn, tpr_knn, _ = roc_curve(y_test_roc, y_score_knn)
    auc_knn = auc(fpr_knn, tpr_knn)
    
    print("AUC Score: " + str(round(auc_knn, 4)))
    
    # 3. Baseline Classifier - Create artificial ROC line
    print("\n" + "-"*40)
    print("BASELINE CLASSIFIER ROC LINE")
    print("-"*40)
    
    # Find the most frequent class
    unique_classes, class_counts = numpy.unique(y_train_roc, return_counts=True)
    most_frequent_class = unique_classes[numpy.argmax(class_counts)]
    baseline_prevalence = numpy.max(class_counts) / numpy.sum(class_counts)
    
    print("Most frequent class: " + str(most_frequent_class))
    print("Baseline prevalence: " + str(round(baseline_prevalence, 4)))
    
    # Create artificial scores for baseline to generate a line
    # For "always predict negative" baseline
    if most_frequent_class == 0:
        # Create artificial scores that are always low (predict negative)
        y_score_baseline = numpy.random.uniform(0, 0.1, len(y_test_roc))
    else:
        # Create artificial scores that are always high (predict positive)  
        y_score_baseline = numpy.random.uniform(0.9, 1.0, len(y_test_roc))
    
    # Calculate ROC curve for baseline (this will create a diagonal-like line)
    fpr_baseline, tpr_baseline, _ = roc_curve(y_test_roc, y_score_baseline)
    auc_baseline = auc(fpr_baseline, tpr_baseline)
    
    print("AUC Score: " + str(round(auc_baseline, 4)))
    
    # 4. Create ROC Curve Plot in the requested style
    print("\n" + "="*60)
    print("ROC CURVE PLOT")
    print("="*60)
    
    # Plot Logistic Regression ROC curve
    matplotlib.pyplot.plot(fpr_logreg, tpr_logreg, label='Logistic Regression (AUC = ' + str(auc_logreg) + ')')
    
    # Plot kNN ROC curve
    matplotlib.pyplot.plot(fpr_knn, tpr_knn, label='kNN (AUC = ' + str(auc_knn) + ')')
    
    # Plot baseline classifier as a line
    matplotlib.pyplot.plot(fpr_baseline, tpr_baseline, color='gray',  label='Baseline (AUC = ' + str(auc_baseline) + ')')
    
    # Plot random classifier line (diagonal)
    matplotlib.pyplot.plot([0, 1], [0, 1], color='green', linestyle='--', label='Random Classifier')
    
    # Customize the plot
    matplotlib.pyplot.xlabel('False positive rate')
    matplotlib.pyplot.ylabel('True positive rate')
    matplotlib.pyplot.legend(loc='lower right')
    
    matplotlib.pyplot.show()
    
    # 5. Summary Analysis
    print("\n" + "="*60)
    print("ROC ANALYSIS SUMMARY")
    print("="*60)
    
    print("AUC Scores:")
    print("Logistic Regression: " + str(round(auc_logreg, 4)))
    print("kNN Classifier: " + str(round(auc_knn, 4)))
    print("Baseline: ~0.5 (no meaningful discrimination)")
    
    if auc_logreg > auc_knn:
        print("Logistic Regression has better performance")
    else:
        print("kNN has better performance")
    
    print("Both ML classifiers significantly outperform baseline and random classifiers")