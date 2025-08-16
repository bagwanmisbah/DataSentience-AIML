Company Profit Classifier

This project is designed to classify companies into "High Profit" or "Low Profit" categories based on their financial and industry attributes. The dataset is derived from the Forbes Global 2000 Companies 2025, which includes leading companies worldwide across diverse industries.

🔍 Problem Statement

The objective is to predict whether a company is highly profitable or not by leveraging its:

Sales ($B)

Assets ($B)

Market Value ($B)

Industry

This is a binary classification problem where the target variable is the profitability status of a company.

🏗️ Approach

Data Preprocessing (preprocess.py)

Load and clean the dataset.

Encode categorical features like Industry.

Create the target column:

Companies with profit above the median profit → High Profit

Companies with profit below or equal to the median → Low Profit

Split the dataset into training and testing sets.

Model Training (train.py)

Train a Random Forest Classifier (chosen for its robustness with both categorical and numerical data).

Save the trained model to disk for reuse.

Evaluate the model on test data with metrics like accuracy and classification report.

Prediction (predict.py)

Load the trained model.

Accept company financial data as input (sales, assets, market value, and industry).

Predict whether the company is High Profit or Low Profit.