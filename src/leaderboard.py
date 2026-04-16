
import os
import csv
import pandas as pd

def save_leaderboard(results):

    print("🏆 Saving leaderboard...")

    # Create folder if not exists
    os.makedirs('leaderboard', exist_ok=True)

    scores_path = 'leaderboard/scores.csv'

    # Write results
    with open(scores_path, 'w', newline='') as f:
        writer = csv.writer(f)

        writer.writerow(['Model Name', 'Accuracy (%)', 'F1-Score (%)'])

        for name, res in results.items():
            writer.writerow([
                name,
                res.get('Accuracy', 0),
                res.get('F1-Score', 0)
            ])

    print(f"✅ Leaderboard saved at: {scores_path}")

    # Show leaderboard
    df = pd.read_csv(scores_path)
    print("\n=== Leaderboard ===")
    print(df.to_string(index=False))

    return df
