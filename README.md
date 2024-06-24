# Anomaly Fraud Detection Project
# Table of Contents
   - Introduction
   - Project Structure
   - Dataset
   - Preprocessing
   - Modeling
   - Evaluation
   - Realme File Integration
   - Results
   - Usage
   - Contributing
   - License
*Introduction*
This project aims to detect fraudulent activities in financial transactions using anomaly detection techniques. By leveraging machine learning models, the system can identify transactions that deviate significantly from normal behavior patterns, thus flagging potential fraud.

*Project Structure*
Anomaly-Fraud-Detection/
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── realme.csv
├── notebooks/
│   ├── EDA.ipynb
│   ├── Model_Training.ipynb
├── src/
│   ├── preprocessing.py
│   ├── modeling.py
│   ├── evaluation.py
├── images/
│   ├── data_distribution.png
│   ├── model_performance.png
├── README.md
├── requirements.txt
├── LICENSE
*Dataset*
The dataset used in this project consists of financial transaction records. Each record contains multiple features that describe the transaction. The dataset is split into training and testing sets.

*Preprocessing*
  Preprocessing steps include:

  - Handling missing values
  - Encoding categorical variables
  - Normalizing numerical features
*Example of preprocessing steps:*
from src.preprocessing import preprocess_data

train_data = preprocess_data('data/train.csv')
test_data = preprocess_data('data/test.csv')
*Modeling*
Various machine learning models are tested for anomaly detection, including:

Isolation Forest
One-Class SVM
Autoencoders
*Example of model training:*
from src.modeling import train_model

model = train_model(train_data)
*Evaluation*
The models are evaluated using metrics such as:

Precision
Recall
F1-Score
*Realme File Integration*
The realme.csv file contains additional transaction data from Realme. This file is integrated into the existing dataset to enhance the model’s ability to detect fraud.

*Adding Realme Data*
Place realme.csv in the data/ directory.
Update the preprocessing script to include realme.csv:
realme_data = preprocess_data('data/realme.csv')
combined_data = pd.concat([train_data, realme_data], ignore_index=True)
Sample Realme Data:

*Results*
The results of the anomaly detection models are visualized using various plots.

*Data Distribution:*

*Model Performance:*

Usage
To run the project, follow these steps:

*Clone the repository:*
git clone https://github.com/yourusername/Anomaly-Fraud-Detection.git
*Install the required dependencies:*
pip install -r requirements.txt
*Run the preprocessing script:*
python src/preprocessing.py
*Train the model:*
python src/modeling.py
*Evaluate the model:*
python src/evaluation.py
*Contributing*
Contributions are welcome! Please submit a pull request or open an issue to discuss changes.
