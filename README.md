# NIDS_Classification_Project
Network Intrusion Detection using UNSW-NB15 dataset

# Intelligent Network Intrusion Detection System (NIDS)
### Machine Learning for Cyber Attack Classification (UNSW-NB15)

## 📌 Project Overview
This project involves the development of an intelligent Network Intrusion Detection System (NIDS) designed to classify network traffic as either **Normal** or **Attack**. Using a subset of the benchmark **UNSW-NB15 dataset**, we implemented a full machine learning pipeline—from Exploratory Data Analysis (EDA) and robust preprocessing to the comparative evaluation of seven different classification models.

## 🛠️ Technical Workflow
The system follows a strictly defined four-step pipeline:

1.  **Exploratory Data Analysis (EDA):** Analyzed class distributions (70/30 split), identified feature skewness in traffic duration and byte counts, and mapped feature correlations.
2.  **Data Preprocessing:**
    * **Encoding:** Categorical variables (`proto`, `state`, `service`) transformed via Label Encoding.
    * **Scaling:** Applied `StandardScaler` to normalize numerical features, ensuring distance-based models (KNN, SVM) perform optimally.
3.  **Model Implementation:** Built and trained 7 models including:
    * **Traditional:** Logistic Regression, Naive Bayes, KNN, SVM.
    * **Tree-Based:** Random Forest, Gradient Boosting.
    * **Deep Learning:** Multi-Layer Perceptron (Basic Deep Neural Network).
4.  **Evaluation:** Models were assessed using Accuracy, Precision, Recall, and the **F2-Score** to prioritize the detection of malicious activity (Recall).

## 📊 Performance Results

| Model | Accuracy | Precision | Recall | **F2 Score** |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | **0.9249** | **0.9930** | **0.8963** | **0.9141** |
| KNN (k=5) | 0.9096 | 0.9899 | 0.8766 | 0.8971 |
| Gradient Boosting | 0.9043 | 0.9920 | 0.8669 | 0.8893 |
| DNN (MLP) | 0.8759 | 0.9907 | 0.8261 | 0.8545 |
| SVM | 0.8602 | 0.9714 | 0.8195 | 0.8459 |

## 💡 Key Insights & Conclusion
* **The Winner:** **Random Forest** outperformed all other models. Its ensemble nature effectively captured the non-linear "if-then" patterns inherent in network traffic data.
* **Metric Strategy:** In a cybersecurity context, we prioritized the **F2-Score**. While Accuracy tells us how often the model is right, the F2-Score ensures we are catching as many attacks as possible (High Recall) while maintaining high Precision.
* **Practical Implications:** The high precision (99%+) of the Random Forest model suggests that this system would be highly reliable in a production environment, as it generates very few false alarms that would otherwise fatigue security analysts.

## 🚀 How to Run
1. Clone the repository.
2. Ensure you have the datasets (`UNSW_NB15_train_40k.csv` and `UNSW_NB15_test_10k.csv`) in the root folder.
3. Install dependencies:
   ```bash
   pip install pandas scikit-learn seaborn matplotlib
4. Run the Analysis : python nids_analysis.py

---




