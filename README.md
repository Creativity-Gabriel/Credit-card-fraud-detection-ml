# Credit-card-fraud-detection-ml
Machine learning project for detecting fraudulent credit card transactions using Logistic Regression and Random Forest.
# Visual Results

### LOGISTIC REGRESSION vs RANDOM FOREST
<img width="1089" height="590" alt="image" src="https://github.com/user-attachments/assets/2d74da17-dcaf-4440-bd3f-adad8b6b970d" />

Key values shown:

•	Logistic Regression:

Precision 0.056, Recall 0.874, F1 0.106, ROC-AUC 0.965

•	Random Forest: Precision 0.945, Recall 0.726, F1 0.821, ROC-AUC 0.939

The above clearly shows the trade-off between catching fraud (Recall) and reducing false alerts (Precision).



### THRESHOLD vs BUSINESS COST

<img width="1089" height="590" alt="image" src="https://github.com/user-attachments/assets/2fc19705-2c1c-471d-a584-66713be0aed9" />


The chart shows how changing the **classification threshold** affects the estimated business cost.

Using the ₦100,000 false-negative and ₦1,000 false-positive assumptions, the lowest tested cost is at a **0.90 threshold: about ₦1.82 million**.

### RANDOM FOREST---TOP TEN FEATURE IMPORTANCE

<img width="1085" height="590" alt="image" src="https://github.com/user-attachments/assets/026948f4-0ca5-4ca7-93d6-230eb8c99aae" />


### CREDIT CARD FRAUD DETECTION--- CLASS DISTRIBUTION

<img width="889" height="590" alt="image" src="https://github.com/user-attachments/assets/dbfb97ae-2c36-435d-a321-4458cb131ff5" />


### Class Distribution — Key takeaway

* **Legitimate transactions:** 283,253 (**99.83%**)
  
* **Fraudulent transactions:** 473 (**0.17%**)

The visual clearly demonstrates the **severe class imbalance** in the dataset, which is why accuracy alone would be misleading and why we focused on **Precision, Recall, F1-score, ROC-AUC, and threshold analysis**.





### PROJECT OVERVIEW
Credit Card Fraud Detection Using Machine Learning


Built an end-to-end credit card fraud detection system using Python and machine learning to identify potentially fraudulent transactions in a highly imbalanced dataset.
The project focused not only on model accuracy, but also on precision, recall, F1-score, ROC-AUC, probability thresholds, business cost analysis, and feature importance.
The goal was to understand the trade-off between detecting more fraudulent transactions and reducing false fraud alerts.
________________________________________
Dataset
Dataset: Credit Card Fraud Detection Dataset

Source: Kaggle — Machine Learning Group, ULB

The original dataset contained 284,807 transactions and 31 columns.

After removing duplicate transactions:

•	283,726 transactions

•	30 input features

•	1 target variable (Class)

•	473 fraud cases

•	283,253 legitimate transactions


The target variable was:

•	0 = Legitimate transaction

•	1 = Fraudulent transaction

The dataset was extremely imbalanced, with fraud representing approximately 0.17% of the transactions.
________________________________________
Technologies Used

•	Python

•	Pandas

•	NumPy

•	Matplotlib

•	Scikit-learn

•	Jupyter Notebook

Machine Learning Algorithms

•	Logistic Regression

•	Random Forest Classifier
________________________________________
Project Workflow

1. Data Cleaning
Performed:

•	Dataset inspection

•	Missing-value analysis

•	Duplicate detection

•	Duplicate removal

•	Feature/target separation

•	Class-distribution analysis

The dataset contained 1,081 duplicate rows, which were removed before modeling.
________________________________________
2. Exploratory Data Analysis
Investigated:

•	Transaction amount distribution

•	Fraud vs legitimate transaction counts

•	Statistical summaries

•	Feature distributions

•	Class imbalance

•	Relationships between variables

A major finding was the extreme class imbalance, making accuracy an unreliable primary evaluation metric.
________________________________________
3. Train/Test Split
   
Used a stratified 80/20 train-test split to preserve the fraud/non-fraud ratio in both datasets.

Training set:

•	226,980 transactions

Test set:

•	56,746 transactions

The test set contained 95 fraud cases.
________________________________________
4. Feature Scaling
   
Applied StandardScaler to the Logistic Regression features.

The scaler was:

•	fitted only on the training data

•	used to transform the test data

This prevented information from the test set from leaking into model training.
________________________________________
Model 1 — Logistic Regression

A Logistic Regression model was trained using:

LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)
class_weight="balanced" was used to address the severe class imbalance.
Performance at the 0.50 threshold

Metric	Result

Precision	5.64%

Recall	87.37%

F1 Score	10.59%

ROC-AUC	~96.5%

Confusion matrix:

[[55262, 1389],

 [   12,   83]]
 
Therefore:

•	True Negatives = 55,262

•	False Positives = 1,389

•	False Negatives = 12

•	True Positives = 83

Interpretation

The Logistic Regression model detected a large proportion of actual fraud cases, but generated many false-positive alerts.

This resulted in:

•	High recall

•	Very low precision
________________________________________
Model 2 — Random Forest

A Random Forest classifier was trained using:
RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

Feature scaling was not required because Random Forest is a tree-based algorithm.

Performance at the 0.50 threshold

Metric	Result

Precision	94.52%

Recall	72.63%

F1 Score	82.14%

ROC-AUC	93.91%

Confusion matrix:

[[56647,    4],

 [   26,   69]]
 
Therefore:

•	True Negatives = 56,647

•	False Positives = 4

•	False Negatives = 26

•	True Positives = 69

Interpretation

Random Forest produced dramatically fewer false-positive alerts while maintaining substantial fraud detection capability.

However, it missed more actual fraud cases than Logistic Regression.
________________________________________
Model Comparison

Metric	Logistic Regression	Random Forest

Precision	5.64%	94.52%

Recall	87.37%	72.63%

F1 Score	10.59%	82.14%

ROC-AUC	~96.5%	93.91%

False Positives	1,389	4

False Negatives	12	26

Key Finding

The two models demonstrated different operating characteristics:

Logistic Regression

•	Higher recall

•	Lower precision

•	More fraud detected

•	Many more false alerts

Random Forest

•	Much higher precision

•	Lower recall

•	Very few false alerts

•	More missed fraud cases

This demonstrates why fraud detection cannot be evaluated using accuracy alone.
________________________________________
Threshold Analysis

Investigated different probability thresholds for Logistic Regression.

Thresholds from 0.10 to 0.90 were tested.

The analysis demonstrated that changing the threshold changes the balance between:

•	Precision

•	Recall

•	False positives

•	False negatives

•	Business cost

For example, with the hypothetical business costs:

•	Missed fraud = ₦500,000

•	False alert = ₦1,000

the calculated cost at different thresholds was:

Threshold	False Positives	False Negatives	Estimated Cost

0.10	10,162	6	₦13,162,000

0.20	5,116	8	₦9,116,000

0.30	3,019	12	₦9,019,000

0.40	1,993	12	₦7,993,000

0.50	1,389	12	₦7,389,000

0.60	970	12	₦6,970,000

0.70	634	12	₦6,634,000

0.80	419	15	₦7,919,000

0.90	222	16	₦8,222,000

Under these hypothetical assumptions and this test set, the lowest calculated cost among the tested thresholds occurred at 0.70.

This demonstrates how a business can choose an operating threshold based on the relative cost of missed fraud versus false alerts.
________________________________________
Feature Importance

Random Forest

The top features identified by the Random Forest were:

Feature	Importance

V14	18.87%

V10	11.77%

V12	10.24%

V17	9.71%

V4	9.32%

V3	6.68%

V11	5.35%

V16	4.49%

V2	3.65%

V9	2.52%

Logistic Regression

For Logistic Regression, feature importance was examined using the absolute values of standardized model coefficients.

Top features included:

Feature	Coefficient

Amount	+1.621

V14	−1.498

V1	+1.394

V12	−1.255

V10	−1.218

V4	+1.208

V5	+0.891

V17	−0.861

V22	+0.663

V20	−0.651

Positive coefficients push predictions toward the fraud class, while negative coefficients push predictions toward the non-fraud class.

Feature importance represents model behavior/association, not proof that a feature causes fraud.
________________________________________
Key Machine Learning Lessons

Through this project, I learned how to:

•	Handle highly imbalanced classification da
tasets

•	Detect and remove duplicate records

•	Perform exploratory data analysis

•	Use stratified train/test splitting

•	Apply feature scaling correctly

•	Prevent data leakage

•	Train Logistic Regression models

•	Train Random Forest models

•	Interpret confusion matrices

•	Calculate precision, recall and F1-score

•	Evaluate models using ROC-AUC

•	Work with prediction probabilities

•	Adjust classification thresholds

•	Analyze false positives and false negatives

•	Translate model performance into hypothetical business costs

•	Interpret Logistic Regression coefficients

•	Interpret Random Forest feature importance

•	Compare different machine learning algorithms
________________________________________
Business Insight
A major lesson from this project is that the best model cannot be determined from a single metric.

For a fraud detection system, the appropriate balance depends on the business consequences of:

•	Missing fraudulent transactions

•	Generating false fraud alerts

•	Customer inconvenience

•	Investigation costs

•	Fraud losses

Therefore, model evaluation should combine technical performance metrics with business requirements.
________________________________________
Limitations

This project uses a public anonymized historical dataset.

Important limitations include:

1.	The dataset does not expose the original meanings of the anonymized V1–V28 variables.
2.	Results on this dataset may not generalize to live banking transactions.
3.	The test set contains only 95 fraud cases, so performance estimates have uncertainty.
4.	The threshold analysis used the test set for exploration; therefore, those threshold results should not be treated as a completely untouched final evaluation.
5.	The business costs used in threshold analysis were hypothetical.
A production system would require additional validation, fresh holdout testing, monitoring, and recalibration as fraud patterns change.
________________________________________
Project Outcome

Built a complete machine learning fraud detection workflow from raw transaction data through:

Data Cleaning → EDA → Feature Engineering/Scaling → Model Training → Evaluation → Threshold Optimization → Business Cost 

Analysis → Feature Importance

The project demonstrated practical understanding of imbalanced classification, model evaluation, probability thresholds, and

business-oriented machine learning decision-making.


