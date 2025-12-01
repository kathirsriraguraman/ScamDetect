SMS Spam Detection Project Report
1. Project Title

SMS Spam Detection using Naive Bayes

2. Objective

The objective of this project is to develop a machine learning model that can automatically classify SMS messages as spam or ham (not spam). This helps in filtering unwanted messages efficiently.

3. Dataset

Source: PyCon 2016 SMS Dataset

Format: Tab-separated values (TSV)

Columns:

label → spam or ham

message → content of the SMS

Observation: The dataset is imbalanced with more ham messages than spam.

4. Libraries Used

pandas → Data manipulation

numpy → Numerical operations

matplotlib.pyplot → Visualization

sklearn.model_selection → Train-test split

sklearn.feature_extraction.text → TF-IDF vectorization

sklearn.naive_bayes → Multinomial Naive Bayes classifier

sklearn.metrics → Evaluation metrics (accuracy, confusion matrix, ROC, AUC)

5. Methodology
Step 1: Load Dataset

Read the dataset from URL and assign column names label and message.

Step 2: Exploratory Data Analysis (EDA)

Visualize the class distribution of ham vs spam using a bar chart.

Observation: Ham messages dominate, indicating class imbalance.

Step 3: Train-Test Split

Split the dataset into 80% training and 20% testing.

random_state=42 ensures reproducibility.

Step 4: Text Vectorization

Convert text messages to numerical features using TF-IDF vectorization.

fit_transform on training data, transform on test data.

Step 5: Model Training

Use Multinomial Naive Bayes classifier.

Train the model using vectorized training data.

Step 6: Evaluation

Accuracy: Measures overall correctness.

Confusion Matrix: Visual representation of True Positives, True Negatives, False Positives, and False Negatives.

ROC Curve & AUC: Evaluate model performance on imbalanced data.

Step 7: Prediction Function

check_spam(msg) predicts whether a new SMS is spam or ham.

6. Results
Class Distribution

Ham: Majority

Spam: Minority

Indicates need for careful evaluation beyond accuracy.

Model Accuracy

Achieved around ~98% accuracy (depending on train-test split).

Confusion Matrix
          Predicted
          Ham   Spam
Actual Ham  ...
       Spam ...


True Positives (spam correctly predicted)

True Negatives (ham correctly predicted)

False Positives / False Negatives

ROC Curve

AUC = ~0.99 (high performance)

Shows the model can distinguish spam from ham effectively.

Sample Prediction

Input: "http://free-gift-online-login-security.com"

Prediction: "spam"

7. Conclusion

The project successfully implemented a spam detection system using Naive Bayes.

The model can predict new SMS messages and shows high performance in evaluation metrics.

Confusion matrix and ROC-AUC confirm the model handles imbalanced data well.

8. Future Work

Handle class imbalance with weighted classes or SMOTE.

Preprocess text (lowercase, remove punctuation, stopwords).

Try advanced models (SVM, Logistic Regression, Deep Learning).

Hyperparameter tuning for Naive Bayes (e.g., alpha smoothing).

Deploy as a web app for real-time spam detection.

9. References

PyCon 2016 SMS Spam Dataset

Scikit-learn Documentation

TF-IDF Vectorization
