# LightGBM-based Credit Card Fraud Detection with Score Band Table

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import lightgbm as lgb
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory
from functools import wraps
import uuid
import time
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    auc, f1_score, precision_score, recall_score
)

app = Flask(__name__)
app.secret_key = 'your_secret_key'

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULT_FOLDER = os.path.join(BASE_DIR, 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

uploaded_data_path = None
predict_df = None

def main():
    try:
        static_path = 'static'
        os.makedirs(static_path, exist_ok=True)

        print("STEP 1: Load Dataset")
        df = pd.read_csv('creditcard.csv')

        print("STEP 2: Scale 'Amount' and 'Time'")
        df['Amount'] = StandardScaler().fit_transform(df[['Amount']])
        df['Time'] = StandardScaler().fit_transform(df[['Time']])

        X = df.drop('Class', axis=1)
        y = df['Class']

        print("STEP 3: Train-Test Split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        print("STEP 4: Apply SMOTE")
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        print("STEP 5: Train LightGBM Model")
        model = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        print("STEP 6: Evaluation")
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("ROC AUC Score:", roc_auc_score(y_test, y_prob))
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)
        print("PRAUC:", pr_auc)

        fraud_scores = y_prob[y_test == 1]
        non_fraud_scores = y_prob[y_test == 0]
        all_scores = np.sort(np.concatenate([fraud_scores, non_fraud_scores]))
        fraud_cdf = np.searchsorted(np.sort(fraud_scores), all_scores, side='right') / len(fraud_scores)
        non_fraud_cdf = np.searchsorted(np.sort(non_fraud_scores), all_scores, side='right') / len(non_fraud_scores)
        ks_stat = np.max(np.abs(fraud_cdf - non_fraud_cdf))
        print("KS Statistic:", ks_stat)

        print("STEP 7: Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(os.path.join(static_path, 'confusion_matrix.png'))
        plt.show()

        print("STEP 8: ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_prob):.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(static_path, 'roc_curve.png'))
        plt.show()

        print("STEP 9: Precision-Recall Curve")
        plt.figure()
        plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.4f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(static_path, 'precision_recall_curve.png'))
        plt.show()

        print("STEP 10: Feature Importance")
        lgb.plot_importance(model, max_num_features=10, importance_type='gain', figsize=(8, 6))
        plt.title("Top 10 Feature Importances")
        plt.tight_layout()
        plt.savefig(os.path.join(static_path, 'feature_importance.png'))
        plt.show()

        print("STEP 11: Score Band Table")
        test_df = pd.DataFrame({'prob': y_prob, 'actual': y_test})
        bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        labels = ['0 - 9', '10 - 19', '20 - 29', '30 - 39', '40 - 49',
                  '50 - 59', '60 - 69', '70 - 79', '80 - 89', '90 - 100']
        ranges = ['[0 - 0.1)', '[0.1 - 0.2)', '[0.2 - 0.3)', '[0.3 - 0.4)', '[0.4 - 0.5)',
                  '[0.5 - 0.6)', '[0.6 - 0.7)', '[0.7 - 0.8)', '[0.8 - 0.9)', '[0.9 - 1]']
        test_df['Score Band'] = pd.cut(test_df['prob'], bins=bins, labels=labels, include_lowest=True)
        test_df['Score Probabilities'] = pd.cut(test_df['prob'], bins=bins, labels=ranges, include_lowest=True)

        band_table = (
            test_df
            .groupby(['Score Probabilities'], observed=False)['actual']
            .value_counts()
            .unstack(fill_value=0)
            .reset_index()
        )
        band_table.rename(columns={0: 'Non Fraud', 1: 'Fraud'}, inplace=True)
        band_table['Score Band'] = band_table['Score Probabilities'].map(dict(zip(ranges, labels)))
        band_table['Total Transaction'] = band_table['Non Fraud'] + band_table['Fraud']
        band_table['Fraud Rate'] = ((band_table['Fraud'] / band_table['Total Transaction']) * 100).round(2).astype(str) + '%'
        band_table = band_table.sort_values(by='Score Probabilities', ascending=False).reset_index(drop=True)
        band_table['Cum Non Fraud'] = band_table['Non Fraud'].cumsum()
        band_table['Cum Fraud'] = band_table['Fraud'].cumsum()
        total_fraud = band_table['Fraud'].sum()
        total_non_fraud = band_table['Non Fraud'].sum()
        band_table['Det Rate'] = (band_table['Cum Fraud'] / total_fraud * 100).round(2)
        band_table['FPR'] = (band_table['Cum Non Fraud'] / total_non_fraud * 100).round(2)

        band_table = band_table[['Score Band', 'Score Probabilities', 'Non Fraud', 'Fraud',
                                 'Total Transaction', 'Fraud Rate', 'Cum Non Fraud',
                                 'Cum Fraud', 'Det Rate', 'FPR']]

        band_table.to_csv("score_band_cleaned.csv", index=False)
        print("\nFinal Score Band Table:\n")
        print(band_table)

        # New Histogram for Figure 2
        fraud_data = test_df[test_df['actual'] == 1]

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        fraud_data['Score Band'].value_counts().sort_index().plot(kind='bar', color='royalblue')
        plt.title('Histogram of Score')
        plt.xlabel('Score')
        plt.ylabel('Fraud')

        plt.subplot(1, 2, 2)
        fraud_data['Score Probabilities'].value_counts().sort_index().plot(kind='bar', color='royalblue')
        plt.title('Histogram of Predicted Probabilities')
        plt.xlabel('Probabilities')
        plt.ylabel('Fraud')

        plt.tight_layout()
        plt.savefig("static/histogram_fraud_predictions.png")
        plt.show()

        # Plot Rank Ordering of Fraud Rates
        fraud_rate_cleaned = band_table.copy()
        fraud_rate_cleaned['Fraud Rate Numeric'] = fraud_rate_cleaned['Fraud Rate'].str.replace('%', '').astype(float)
        plt.figure(figsize=(8, 5))
        plt.plot(fraud_rate_cleaned['Score Band'], fraud_rate_cleaned['Fraud Rate Numeric'], marker='o')
        plt.title('Rank ordering of fraud rates')
        plt.xlabel('Score')
        plt.ylabel('Fraud Rate')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(static_path, 'rank_ordering_fraud_rates.png'))
        plt.show()

        print("\nSTEP 8: Generating Histogram of Score by Fraud Tag...")

        score_df = pd.DataFrame({'prob': y_prob, 'actual': y_test})
        score_df['score'] = (score_df['prob'] * 100).astype(int)

        fraud_0 = score_df[score_df['actual'] == 0]
        fraud_1 = score_df[score_df['actual'] == 1]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # Removed sharey=True

        bins = list(range(0, 102, 3))

        # Fraud=0
        axes[0].hist(fraud_0['score'], bins=bins, edgecolor='black')
        axes[0].set_title('Fraud=0')
        axes[0].set_xlabel('Scores')
        axes[0].set_ylabel('Non Fraud')
        axes[0].set_xticks(bins)
        axes[0].tick_params(axis='x', rotation=90)

        # Fraud=1
        axes[1].hist(fraud_1['score'], bins=bins, edgecolor='black', color='darkorange')
        axes[1].set_title('Fraud=1')
        axes[1].set_xlabel('Scores')
        axes[1].set_ylabel('Fraud')
        axes[1].set_xticks(bins)
        axes[1].tick_params(axis='x', rotation=90)

        plt.tight_layout()
        plt.savefig(os.path.join(static_path, 'score_by_fraud_tag.png'))
        plt.show()

        print("Histogram saved: score_by_fraud_tag.png")

        # Bar chart
        fraud_counts = [sum(y_test == 0), sum(y_test == 1)]
        plt.figure(figsize=(6, 4))
        bars = plt.bar(['Not Fraud', 'Fraud'], fraud_counts, color=['skyblue', 'salmon'])

        # Add text labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 50, f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        frauds = sum(y_pred)
        legit = len(y_pred) - frauds

        plt.title('Fraud vs Not Fraud Transactions')
        plt.ylabel('Number of Transactions')
        plt.xlabel('Class')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(static_path, 'fraud_vs_notfraud.png'))  # Optional: save the figure
        plt.show()

        # Pie Chart: Fraud vs Not Fraud
        fig, ax = plt.subplots()
        ax.pie([frauds, legit],
        labels=['Fraudulent', 'Not Fraudulent'],
        autopct='%1.1f%%',
        startangle=90,
        colors=['red', 'green'])
        ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
        plt.title('Fraud vs Not Fraud Transactions')
        plt.tight_layout()
        plt.savefig(os.path.join('static', 'fraud_pie_chart.png'))
        plt.show()

        print("STEP 12: Save Model and Scalers")
        joblib.dump(model, 'fraud_model.pkl')
        joblib.dump(StandardScaler().fit(df[['Amount']]), 'amount_scaler.pkl')
        joblib.dump(StandardScaler().fit(df[['Time']]), 'time_scaler.pkl')
        print("Model and scalers saved")

    except Exception as e:
        print("An error occurred:", e)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            flash("Please log in first.")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == 'admin' and password == 'password':
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('admin'))
        elif username == 'user' and password == 'password':
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('home'))

        flash("Invalid credentials. Please try again.")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.")
    return redirect(url_for('login'))

@app.route('/')
@login_required
def home():
    return render_template('home.html')

@app.route('/index', methods=['GET', 'POST'])
@login_required
def index():
    global uploaded_data_path
    message = ""
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            filename = f"uploaded_{uuid.uuid4().hex}.csv"
            uploaded_data_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(uploaded_data_path)
            message = "File uploaded successfully."
        else:
            flash("Please upload a valid CSV file")
    return render_template('index.html', table=None, uploaded_data=True, message=message, page=1, total_pages=1, max=max, min=min)

@app.route('/view_data')
@login_required
def view_data():
    global uploaded_data_path
    if not uploaded_data_path or not os.path.exists(uploaded_data_path):
        flash("No file uploaded yet.")
        return redirect(url_for('index'))

    df = pd.read_csv(uploaded_data_path)
    df.drop(columns=[col for col in ['Class', 'Predicted', 'Fraud_Probability', 'Result'] if col in df.columns], inplace=True)
    page = int(request.args.get('page', 1))
    per_page = 100
    start, end = (page - 1) * per_page, page * per_page
    total_pages = (len(df) + per_page - 1) // per_page
    table_html = df.iloc[start:end].to_html(classes='table table-bordered', index=False)
    return render_template('index.html', table=table_html, uploaded_data=True, message="Uploaded data:", page=page, total_pages=total_pages, max=max, min=min, pagination_endpoint='view_data')

@app.route('/predict_page', methods=['GET', 'POST'])
@login_required
def predict_page():
    global uploaded_data_path, predict_df
    message = ""
    page = int(request.args.get('page', 1))
    per_page = 100

    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            print("[DEBUG] File received:", file.filename)
            filename = f"uploaded_{uuid.uuid4().hex}.csv"
            uploaded_data_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(uploaded_data_path)
            print("[DEBUG] File saved to:", uploaded_data_path)

            df = pd.read_csv(uploaded_data_path)
            print("[DEBUG] Uploaded columns:", df.columns.tolist())

            required = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
            X = df[required].copy()

            print("[DEBUG] Model loaded:", model is not None)
            print("[DEBUG] Amount scaler loaded:", amount_scaler is not None)
            print("[DEBUG] Time scaler loaded:", time_scaler is not None)

            X['Time'] = time_scaler.transform(X[['Time']])
            X['Amount'] = amount_scaler.transform(X[['Amount']])

            df['Predicted'] = model.predict(X)
            df['Fraud_Probability'] = model.predict_proba(X)[:, 1]
            df['Result'] = df['Predicted'].map({1: 'Fraudulent', 0: 'Not Fraudulent'})

            result_filename = f"result_{uuid.uuid4().hex}.csv"
            result_path = os.path.join(RESULT_FOLDER, result_filename)

            print("[DEBUG] Writing prediction result to:", result_path)
            print("[DEBUG] Does folder exist?", os.path.exists(RESULT_FOLDER))

            df.to_csv(result_path, index=False)
            print("[DEBUG] Saved result to", result_path)
            print("[DEBUG] Current files in results/:", os.listdir(RESULT_FOLDER))

            predict_df = df.copy()
            message = "Prediction completed."

    if predict_df is not None:
        total_pages = (len(predict_df) + per_page - 1) // per_page
        page = min(page, total_pages)
        start, end = (page - 1) * per_page, page * per_page
        view = predict_df.iloc[start:end]

        def style_row(row):
            return [f'<td style="color:red">{v}</td>' if row['Result'] == 'Fraudulent' else f'<td>{v}</td>' for v in row]

        rows = view.apply(style_row, axis=1)
        header = ''.join(f'<th>{col}</th>' for col in view.columns)
        table_html = f"<table class='table table-bordered'><thead><tr>{header}</tr></thead><tbody>"
        for row in rows:
            table_html += '<tr>' + ''.join(row) + '</tr>'
        table_html += '</tbody></table>'

        return render_template('predict_page.html', table=table_html, page=page, total_pages=total_pages, message=message, pagination_endpoint='predict_page')

    return render_template('predict_page.html', table=None, page=1, total_pages=1, message=message, pagination_endpoint='predict_page')

@app.route('/manual')
@login_required
def manual_entry():
    global uploaded_data_path
    table_html, total_pages = None, 1
    page = int(request.args.get('page', 1))
    per_page = 100

    if uploaded_data_path and os.path.exists(uploaded_data_path):
        df = pd.read_csv(uploaded_data_path)
        df.drop(columns=[c for c in ['Class', 'Result'] if c in df.columns], inplace=True, errors='ignore')
        total_pages = (len(df) + per_page - 1) // per_page
        start, end = (page - 1) * per_page, page * per_page
        table_html = df.iloc[start:end].to_html(classes='table table-bordered table-hover table-sm', index=False)

    return render_template('manual.html', table=table_html, page=page, total_pages=total_pages)

@app.route('/manual_predict', methods=['POST'])
@login_required
def manual_predict():
    try:
        time_val = float(request.form['Time'])
        amount_val = float(request.form['Amount'])
        v_features = [float(request.form[f'V{i}']) for i in range(1, 29)]

        time_scaled = time_scaler.transform([[time_val]])[0][0]
        amount_scaled = amount_scaler.transform([[amount_val]])[0][0]
        input_vector = [time_scaled] + v_features + [amount_scaled]

        proba = model.predict_proba([input_vector])[0][1]
        result = 'Fraudulent Transaction' if proba >= 0.5 else 'Not Fraudulent'

        # ✅ SAVE TO FILE
        manual_df = pd.DataFrame([{
            'Time': time_val,
            **{f'V{i}': v_features[i - 1] for i in range(1, 29)},
            'Amount': amount_val,
            'Predicted': 1 if result == 'Fraudulent Transaction' else 0,
            'Fraud_Probability': round(proba, 4),
            'Result': result
        }])

        manual_path = os.path.join(RESULT_FOLDER, 'manual_predictions.csv')
        if os.path.exists(manual_path):
            manual_df.to_csv(manual_path, mode='a', header=False, index=False)
        else:
            manual_df.to_csv(manual_path, index=False)

        print(f"[INFO] Manual prediction saved to {manual_path}")

        # Optional: Load and display uploaded file (as before)
        table_html, total_pages, page = None, 1, 1
        if uploaded_data_path and os.path.exists(uploaded_data_path):
            df = pd.read_csv(uploaded_data_path)
            df.drop(columns=[col for col in ['Class', 'Predicted', 'Fraud_Probability', 'Result'] if col in df.columns], inplace=True)
            expected_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
            df = df[[col for col in expected_cols if col in df.columns]]
            per_page = 100
            total_pages = (len(df) + per_page - 1) // per_page
            paginated_df = df.iloc[0:100]
            table_html = paginated_df.to_html(classes='table table-bordered table-hover table-sm', index=False)

        return render_template('manual.html',
                               result=result,
                               probability=f"{proba:.2f}",
                               table=table_html,
                               total_pages=total_pages,
                               page=page)
    except Exception as e:
        print("Manual predict error:", e)
        return render_template('manual.html',
                               result="Error during prediction",
                               probability="N/A",
                               table=None,
                               total_pages=1,
                               page=1)

@app.route('/admin')
@login_required
def admin():
    if session.get('username') != 'admin':
        flash("Access denied. Admins only.")
        return redirect(url_for('home'))

    uploaded_info = []
    for fname in os.listdir(UPLOAD_FOLDER):
        full_path = os.path.join(UPLOAD_FOLDER, fname)
        if os.path.isfile(full_path):
            uploaded_info.append({
                'filename': fname,
                'uploaded_on': time.ctime(os.path.getctime(full_path))
            })

    total_files = len(uploaded_info)
    print("[DEBUG] RESULT_FOLDER:", RESULT_FOLDER)
    print("[DEBUG] Files in RESULT_FOLDER:", os.listdir(RESULT_FOLDER))

    result_files = [f for f in os.listdir(RESULT_FOLDER) if f.endswith('.csv')]
    manual_preview = None
    manual_file = os.path.join(RESULT_FOLDER, 'manual_predictions.csv')
    if os.path.exists(manual_file):
        manual_df = pd.read_csv(manual_file)
        manual_preview = manual_df.tail(5).to_html(classes='table table-bordered table-sm table-hover', index=False)

    total_results = len(result_files)

    return render_template('admin.html',
                       uploaded_info=uploaded_info,
                       total_files=total_files,
                       total_results=total_results,
                       manual_preview=manual_preview)

@app.route('/uploads/<filename>')
@login_required
def download_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True, manual_preview=manual_preview)

@app.route('/delete_file', methods=['POST'])
@login_required
def delete_file():
    if session.get('username') != 'admin':
        flash("Only admin can delete files.")
        return redirect(url_for('admin'))

    filename = request.form.get('filename')
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    if filename and os.path.exists(file_path):
        os.remove(file_path)
        flash(f"File '{filename}' deleted successfully.")
    else:
        flash("File not found or already deleted.")

    return redirect(url_for('admin'))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        print(">>> Training mode enabled.")
        main()
        print(">>> Training completed. Exiting app.")
    else:
        print(">>> Starting Flask web app.")
        model = joblib.load('fraud_model.pkl')
        amount_scaler = joblib.load('amount_scaler.pkl')
        time_scaler = joblib.load('time_scaler.pkl')
        app.run(debug=True)
