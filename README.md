🏦 Customer Transaction Prediction

Internship Project | Machine Learning | Classification

🎯 Objective

The goal of this project is to predict whether a customer will make a financial transaction in the future.
The dataset is anonymized and contains 200 unnamed features, along with:

Column	Meaning
ID_code	Unique customer ID
target	0 = No Transaction, 1 = Will Make Transaction
📂 Dataset

The dataset is a binary classification dataset with:

200 anonymized numerical input features

Highly imbalanced target classes
(Much more 0 → No Transaction customers than 1 → Transaction customers)

Because features are anonymized, full EDA is not meaningful, and thus was skipped.

🧠 Approach
1️⃣ Load & Prepare Data

Import dataset using pandas

Separate features (X) and target variable (y)

Perform train-test split (80% training, 20% testing)

2️⃣ Handle Class Imbalance

The dataset has more class 0 than class 1, causing the model to learn mostly on “No Transaction” cases.

✅ We used class_weight="balanced" in the final model to improve performance for class 1 (important customers).

3️⃣ Model Training & Comparison

We trained multiple models and compared them:

Model	Accuracy	Remarks
Logistic Regression	Medium	Weak on minority class
Random Forest	High	Good, robust
Gradient Boosting	Best	Strong performance, chosen model
4️⃣ Selected Final Model
RandomForestClassifier(n_estimators=300, class_weight="balanced")


This model improves recall for important transaction customers while maintaining strong accuracy.

📊 Evaluation Results
Metric	Score
Accuracy	91%
Precision (Class 1)	0.70
Recall (Class 1)	0.26
F1 Score (Class 1)	0.38
Confusion Matrix
	Predicted No	Predicted Yes
Actual No (0)	44292	588
Actual Yes (1)	3780	1340

Interpretation:

Model is excellent at identifying customers who will not make transactions.

Detecting transaction customers is harder due to imbalance, improved using class weighting.

🏁 Conclusion

The model can accurately identify customers who are unlikely to make a transaction.

After applying class balancing, the model performed significantly better on identifying future transaction customers.

This model can be used by the bank to:

Target marketing campaigns

Increase engagement

Focus offers on potential active customers

🧱 Challenges Faced & Solutions
Challenge	Solution	Reasoning
No feature names (anonymized data)	Skipped deep EDA	EDA would not provide meaningful business insights
Highly imbalanced classes	Used class_weight="balanced"	Forces model to pay more attention to minority class (transactions)
Many features (200)	Used tree-based models	They handle high-dimensional numeric data well
🛠 Tech Stack

Python

Pandas, NumPy

Scikit-learn

Seaborn, Matplotlib

Jupyter Notebook

📁 Project Structure
├── customer_transaction_prediction.ipynb   # Full project notebook
├── README.md                               # Documentation (this file)
└── dataset.csv                              # Training dataset
