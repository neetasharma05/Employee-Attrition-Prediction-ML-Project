
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc

def evaluate_models(results, y_test, X_test):

    print("📊 Running evaluation...")

    # Confusion Matrix
    for name, res in results.items():
        cm = confusion_matrix(y_test, res['y_pred'])

        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(name)
        plt.show()

    # ROC Curve
    plt.figure()

    for name, res in results.items():
        model = res['model']

        try:
            proba = model.predict_proba(X_test)[:, 1]
        except:
            continue

        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{name} ({roc_auc:.2f})")

    plt.legend()
    plt.title("ROC Curve")
    plt.show()

    # Results Table
    df = pd.DataFrame([
        {'Model': name, 'Accuracy': res['Accuracy']}
        for name, res in results.items()
    ])

    print("\nModel Results:")
    print(df)

    print("✅ Evaluation complete!")
