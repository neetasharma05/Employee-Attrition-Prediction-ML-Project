
import pandas as pd

def save_results_summary(results):

    print("📊 Saving results summary...")

    summary = []

    for name, res in results.items():
        summary.append({
            'Model': name,
            'Accuracy': res.get('Accuracy', 0),
            'Precision': res.get('Precision', 0),
            'Recall': res.get('Recall', 0),
            'F1-Score': res.get('F1-Score', 0)
        })

    df = pd.DataFrame(summary)

    df.to_csv('results_summary.csv', index=False)

    print("✅ Results saved → results_summary.csv")
    print(df)

    return df
