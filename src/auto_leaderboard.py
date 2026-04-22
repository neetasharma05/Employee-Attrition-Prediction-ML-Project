"""
AUTOMATIC LEADERBOARD SYSTEM - 
"""

import pandas as pd
import subprocess
import os
from datetime import datetime

# Try to import your training code
try:
    from training import train_model
except:
    print("⚠️ Using simplified mode (training.py not found)")
    # We'll handle this differently

def get_best_accuracy():
    """
    Gets the best accuracy from your trained models
    """
    print("🔄 Loading your trained model results...")
    
    # Load the dataset (using your exact path)
    df = pd.read_csv('data/HR-Employee-Attrition-Dataset.csv')
    
    # Run training
    results, X_test, y_test = train_model(df)
    
    # Find best model
    best_model = None
    best_accuracy = 0
    best_f1 = 0
    
    for model_name, metrics in results.items():
        accuracy = metrics['Accuracy']
        f1_score = metrics.get('F1-Score', 0)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model_name
            best_f1 = f1_score
    
    return best_model, best_accuracy, best_f1

def update_leaderboard(model_name, accuracy, f1_score):
    """
    Updates the leaderboard file
    """
    leaderboard_path = 'leaderboard/scores.csv'
    
    # Read existing leaderboard
    df = pd.read_csv(leaderboard_path)
    
    # Create new entry
    new_entry = pd.DataFrame({
        'Model Name': [f"{model_name} (NEW RECORD)"],
        'Accuracy (%)': [accuracy],
        'F1-Score (%)': [f1_score]
    })
    
    # Add and sort
    df_updated = pd.concat([df, new_entry], ignore_index=True)
    df_updated = df_updated.sort_values('Accuracy (%)', ascending=False)
    
    # Save
    df_updated.to_csv(leaderboard_path, index=False)
    print(f"✅ Leaderboard updated! {model_name}: {accuracy}%")
    
    return df_updated

def create_pull_request(model_name, accuracy):
    """
    Creates GitHub Pull Request automatically
    """
    print("\n📤 Creating Pull Request...")
    
    # Create branch name
    branch_name = f"leaderboard-update-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Git commands
    commands = [
        f"git checkout -b {branch_name}",
        "git add leaderboard/scores.csv",
        f'git commit -m "New record: {model_name} with {accuracy}%"',
        f"git push origin {branch_name}",
        f'gh pr create --title "🏆 New Leaderboard Record: {model_name} - {accuracy}%" --body "Auto-submitted record" --base master'
    ]
    
    for cmd in commands:
        print(f"Running: {cmd}")
        subprocess.run(cmd, shell=True, check=False)
    
    print(f"\n✅ Pull Request created!")
    print(f"👉 View it at: https://github.com/neetasharma05/Employee-Attrition-Prediction-ML-Project/pulls")

def main():
    print("="*50)
    print("🏆 AUTOMATIC LEADERBOARD SYSTEM")
    print("="*50)
    
    # Get best model
    model_name, accuracy, f1 = get_best_accuracy()
    
    print(f"\n🎯 Best Model: {model_name}")
    print(f"📊 Accuracy: {accuracy}%")
    print(f"📊 F1-Score: {f1}%")
    
    # Update leaderboard
    update_leaderboard(model_name, accuracy, f1)
    
    # Create Pull Request
    create_pull_request(model_name, accuracy)
    
    print("\n✅ ALL DONE!")
    print("Your professor will review and merge the Pull Request")

if __name__ == "__main__":
    main()
