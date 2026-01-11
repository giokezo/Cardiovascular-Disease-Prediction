import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score,
                             f1_score,fbeta_score,roc_auc_score, roc_curve, auc)
from sklearn.tree import plot_tree
from sklearn.inspection import permutation_importance

# Define the function to evaluate the models predictions
def evaluate(y_true,y_pred):
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))

# Define the function to plot ROC curves of a model
def plot_roc(y_true, y_prob, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(9,6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0,1], [0,1], '--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    plt.legend()
    plt.show()

# Define the function to plot the confusion matrix of the models predictions
def plot_conf_matrix(y_true, y_pred, labels=None, title="Confusion Matrix", cmap="Blues"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=cmap, values_format='d')
    plt.title(title)
    plt.grid(False)
    plt.show()

# Define the function to plot feature importances of a model
def plot_feature_importances(model, df):
    coef = model.coef_[0]
    coef_df = pd.DataFrame({
        "feature": df.columns,
        "coef": coef,
        "abs_coef": np.abs(coef)
    }).sort_values("abs_coef", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(coef_df["feature"], coef_df["abs_coef"])
    plt.xlabel("Coefficient Value")
    plt.title("Signed Coefficients")
    plt.gca().invert_yaxis()
    plt.show()

# Define a function to plot impurity feature importances for tree models
def impurity_importance_plot(model, X_train, title="Tree Model"):
    feature_names = X_train.columns

    impurity_importance = pd.Series(
        model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # Bar plot
    impurity_importance.head(15).plot(kind="bar", color="steelblue", ax=ax1)
    ax1.set_title(f"{title} - Impurity-Based Feature Importance", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Importance", fontsize=11)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)

    # Table
    top_10 = impurity_importance.head(10)
    table_data = [[feat, f"{val:.4f}"] for feat, val in top_10.items()]
    table = ax2.table(cellText=table_data, colLabels=['Feature', 'Importance'],
                      cellLoc='left', loc='center', colWidths=[0.7, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(2):
            table[(i, j)].set_facecolor('#E7E6E6' if i % 2 == 0 else 'white')

    ax2.axis('off')
    ax2.set_title("Top 10 Features", fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.show()

# Define a function to plot permutation importances for tree models
def permutation_importance_plot(model, X_train, y_train, X_test=None, y_test=None, title="Tree Model"):
    feature_names = X_train.columns

    # Use test set if available, otherwise train
    X_explain_df = X_test if X_test is not None else X_train
    y_explain = y_test if y_test is not None else y_train

    perm = permutation_importance(
        estimator=model,
        X=X_explain_df,
        y=y_explain,
        n_repeats=15,
        random_state=42,
        n_jobs=-1
    )

    perm_importance = pd.Series(
        perm.importances_mean,
        index=feature_names
    ).sort_values(ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # Bar plot
    perm_importance.head(15).plot(kind="bar", color="darkorange", ax=ax1)
    ax1.set_title(f"{title} - Permutation Feature Importance", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Importance Drop", fontsize=11)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)

    # Table
    top_10 = perm_importance.head(10)
    table_data = [[feat, f"{val:.4f}"] for feat, val in top_10.items()]
    table = ax2.table(cellText=table_data, colLabels=['Feature', 'Importance'],
                      cellLoc='left', loc='center', colWidths=[0.7, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#ED7D31')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(2):
            table[(i, j)].set_facecolor('#FBE5D6' if i % 2 == 0 else 'white')

    ax2.axis('off')
    ax2.set_title("Top 10 Features", fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.show()