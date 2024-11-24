from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Initialize Flask App
app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open('kmeans_model.pkl', 'rb'))


# Function to load and clean data
def load_clean_data(file_path):
    df = pd.read_csv(file_path, sep=",", encoding="ISO-8859-1", header=0)
    df['CustomerID'] = df['CustomerID'].astype(str)

    # Compute RFM metrics
    df['Monetary'] = df['Quantity'] * df['UnitPrice']
    rfm_m = df.groupby('CustomerID')['Monetary'].sum().reset_index()
    rfm_m.columns = ['CustomerID', 'Monetary']

    rfm_f = df.groupby('CustomerID')['InvoiceNo'].count().reset_index()
    rfm_f.columns = ['CustomerID', 'Frequency']

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d-%m-%Y %H:%M')
    max_date = df['InvoiceDate'].max()
    df['Diff'] = max_date - df['InvoiceDate']

    rfm_r = df.groupby('CustomerID')['Diff'].min().reset_index()
    rfm_r['Recency'] = rfm_r['Diff'].dt.days
    rfm_r = rfm_r.drop('Diff', axis=1)

    # Merge RFM metrics
    rfm = pd.merge(rfm_m, rfm_f, on='CustomerID', how='inner')
    rfm = pd.merge(rfm, rfm_r, on='CustomerID', how='inner')

    # Remove outliers
    for col in ['Monetary', 'Frequency', 'Recency']:
        q1 = rfm[col].quantile(0.25)
        q3 = rfm[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        rfm = rfm[(rfm[col] > lower_bound) & (rfm[col] < upper_bound)]

    return rfm


# Function to preprocess data for clustering
def preprocess_data(rfm):
    rfm_df = rfm[['Monetary', 'Frequency', 'Recency']]
    scaler = StandardScaler()
    rfm_df_scaled = scaler.fit_transform(rfm_df)
    rfm_df_scaled = pd.DataFrame(rfm_df_scaled, columns=rfm_df.columns)
    return rfm_df_scaled


# Home route
@app.route('/')
def home():
    return render_template('index.html')


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Save uploaded file
        file = request.files['file']
        file_path = os.path.join(os.getcwd(), file.filename)
        file.save(file_path)

        # Load and preprocess data
        rfm = load_clean_data(file_path)
        rfm_scaled = preprocess_data(rfm)

        # Predict clusters
        results_df = model.predict(rfm_scaled)
        rfm['Cluster_Id'] = results_df

        # Generate and save plots
        img_dir = os.path.join('static')
        os.makedirs(img_dir, exist_ok=True)

        monetary_img_path = os.path.join(img_dir, 'ClusterId_Monetary.png')
        sns.stripplot(x='Cluster_Id', y='Monetary', data=rfm)
        plt.savefig(monetary_img_path)
        plt.close()

        frequency_img_path = os.path.join(img_dir, 'ClusterId_Frequency.png')
        sns.stripplot(x='Cluster_Id', y='Frequency', data=rfm)
        plt.savefig(frequency_img_path)
        plt.close()

        recency_img_path = os.path.join(img_dir, 'ClusterId_Recency.png')
        sns.stripplot(x='Cluster_Id', y='Recency', data=rfm)
        plt.savefig(recency_img_path)
        plt.close()

        # Return image paths in response
        response = {
            'monetary_img': monetary_img_path,
            'frequency_img': frequency_img_path,
            'recency_img': recency_img_path
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
