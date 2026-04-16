
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_model(df):

    print("🚀 Training started...")

    # ========================
    # Encoding
    # ========================
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    le = LabelEncoder()

    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    print("✅ Encoding done")

    # ========================
    # Split
    # ========================
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # ========================
    # Scaling
    # ========================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("✅ Scaling done")

    # ========================
    # Models
    # ========================
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5,
            random_state=42, eval_metric='logloss'
        )
    }

    results = {}

    # ========================
    # Training Loop
    # ========================
    for name, model in models.items():

        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        results[name] = {
            'Accuracy': round(acc * 100, 2),
            'Precision': round(report['1']['precision'] * 100, 2),
            'Recall': round(report['1']['recall'] * 100, 2),
            'F1-Score': round(report['1']['f1-score'] * 100, 2),
            'model': model,
            'y_pred': y_pred
        }

        print(f"{name} → Accuracy: {acc*100:.2f}%")

    print("\n✅ All models trained!")

    # ========================
    # XGBoost Tuning
    # ========================
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6],
        'learning_rate': [0.05, 0.1]
    }

    grid_search = GridSearchCV(
        XGBClassifier(eval_metric='logloss'),
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_xgb = grid_search.best_estimator_
    y_pred_xgb = best_xgb.predict(X_test)

    acc = accuracy_score(y_test, y_pred_xgb)

    results['XGBoost (Tuned)'] = {
        'Accuracy': round(acc * 100, 2),
        'model': best_xgb,
        'y_pred': y_pred_xgb
    }

    print("🔥 XGBoost tuning complete!")

    return results, X_test, y_test
