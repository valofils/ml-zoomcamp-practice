# ML Engineering Practice

End-to-end machine learning engineering project built module by module,
inspired by the ML Zoomcamp curriculum. All code is written in Python (`.py` scripts only).

## Stack

- Python 3.11+
- scikit-learn, XGBoost, TensorFlow / PyTorch
- FastAPI — model serving
- BentoML — ML deployment and packaging
- Docker — containerization
- pandas, NumPy, matplotlib, seaborn

## Structure

| Module | Topic |
|--------|-------|
| `01-intro/` | ML fundamentals and environment setup |
| `02-regression/` | Linear regression, EDA, feature engineering |
| `03-classification/` | Logistic regression, encoding, feature selection |
| `04-evaluation/` | Metrics: ROC-AUC, precision, recall, cross-validation |
| `05-deployment/` | Model serialization, FastAPI, Docker |
| `06-trees/` | Decision trees, Random Forest, XGBoost |
| `08-deep-learning/` | Neural networks, CNNs, transfer learning |
| `09-serverless/` | AWS Lambda deployment |
| `10-serving/` | BentoML model serving and packaging |

## Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/ml-zoomcamp-practice.git
cd ml-zoomcamp-practice

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

Each module folder is self-contained. Navigate into a module and run scripts individually:

```bash
cd 02-regression
python eda.py
python train.py
```

## TTableau Dashboard
https://public.tableau.com/app/profile/mariel.ambratis.fils.andrianavalondrahona/viz/GlobalFoodPriceMonitorWFP2015-2024/Dashboard1?publish=yes
