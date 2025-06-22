# 🧠 FactScan – Fake News Detection using Machine Learning

**FactScan** is a machine learning-based project focused on detecting fake news through rigorous research, experimentation with various classification algorithms, and comparative analysis using multiple performance metrics.

This project explores and evaluates the effectiveness of different ML models on fake news datasets and visualizes how well each model performs using standard evaluation metrics like accuracy, precision, recall, F1 score, and MCC.

---

## 📌 Project Objective

To explore various machine learning models for fake news detection by:
- Preprocessing news data using NLP techniques
- Training and testing multiple ML algorithms
- Comparing their performance using visual metrics
- Recommending the most suitable model for production-level deployment

---

## 🔍 Key Highlights

- ✅ In-depth research on fake news detection techniques
- 🔁 Multiple ML models trained and compared
- 📊 Detailed performance evaluation and visualization
- 📂 Clean, modular code and organized notebooks
- 📝 Human-readable documentation and reporting

---

## 🛠️ Technologies & Tools Used

| Category            | Tools / Libraries                           |
|---------------------|---------------------------------------------|
| Language            | Python                                      |
| ML Models           | Naive Bayes, K-Nearest Neighbors (KNN), Logistic Regression, SVM |
| NLP Preprocessing   | NLTK, Regex, Stemming, Stopword Removal     |
| Data Visualization  | Matplotlib, Seaborn                         |
| Dataset Source      | Kaggle Fake News Dataset                    |
| Development Tools   | Jupyter Notebook, VS Code                   |

---

## 🧪 Models Explored

1. **Multinomial Naive Bayes**
2. **Logistic Regression**
3. **Support Vector Machine (SVM)**
4. **K-Nearest Neighbors (KNN)**

Each model was trained and tested on a cleaned version of the fake news dataset and evaluated using standard classification metrics.

---

## 🔍 Text Preprocessing Steps

- Lowercasing all text
- Removing punctuation and digits
- Removing stopwords
- Stemming using `PorterStemmer`
- Tokenization using NLTK

---

## 📊 Evaluation Metrics

We used the following metrics to compare models:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **MCC (Matthews Correlation Coefficient)**

### 📈 Visualization Example (KNN over different values of `k`):
- Accuracy vs k
- Precision vs k
- Recall vs k
- F1 Score vs k
- MCC vs k

All visualizations were done using `matplotlib`.

