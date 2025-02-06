# AI STOCKER

## Project Overview
This project aims to create a **high-frequency stock price prediction model** using **Artificial Intelligence (AI)** and **Deep Learning (DL)** techniques to forecast stock prices. The model improves upon traditional daily predictions by offering **timely, fine-grained insights** to support more agile trading decisions.

---

## Methodology
1. **Data Collection**: Gather day-based stock price data, such as open, high, low, and closed.
2. **Data Preprocessing**: Clean and normalize data to ensure it is suitable for model input.
3. **Feature Engineering**: Extract key features like trading volume and volatility measures.
4. **Model Design**:
   - Implement LSTM to capture long-term dependencies.
   - Use CNN to identify short-term patterns and trends.
   - Combine CNN-LSTM for an integrated approach to enhance accuracy.
5. **Training and Evaluation**:
   - Split the data into training, validation, and testing sets.

---

## Expected Outcome
The project will deliver an accurate, **real-time stock price prediction model** capable of day-wise predictions. Key benefits include:
- Improved prediction accuracy for both short-term and long-term market movements.
- Granular, daily insights to help investors make timely and informed trading decisions.
- Potential integration into trading platforms or advisory tools for practical applications.

---

## Project Highlights
- **Granularity**: Day-wise predictions for fine-grained insights.
- **Hybrid Approach**: Combines LSTM and CNN for enhanced prediction accuracy.
- **Automation**: AI and ML streamline data processing and feature selection.
- **Real-Time Capability**: Supports agile and responsive trading decisions.

---

## Technologies Used
- **Python**
- **TensorFlow/Keras** (Deep Learning frameworks)
- **Pandas, NumPy** (Data manipulation)
- **Scikit-learn** (Machine Learning utilities)
- **Matplotlib/Seaborn** (Visualization tools)
- **Jupyter Notebook** (Model development environment)

---

## How to Run the Project
## Django Setup Commands

### Check Django version:
```sh
py -m django --version
```

### Create a virtual environment:
```sh
python -m venv env_name
```

### Create a Django project:
```sh
django-admin startproject project_name .
```

### Create a Django app:
```sh
django-admin startapp app_name
```

### Run the server:
```sh
python manage.py runserver
```

---

## Steps to Set Up the Project

### 1. Clone this repository:
```sh
git clone https://github.com/shivannadm/AI_Stocker
cd AI-Forecasting-for-Investor-Decision
```

### 2. Install required dependencies:
```sh
pip install -r requirements.txt
```

### 3. Run Jupyter Notebook to train and evaluate the model:
```sh
jupyter notebook
```

### 4. Follow the instructions in the notebook to:
- Preprocess the data
- Train the models
- Generate predictions

---

# Project Structure

## Directories
- `.vscode/` - Configuration files for VS Code.
- `aiApp/` - AI-related Django app.
- `aiProj/` - Main project directory.
- `modelApp/` - Another Django app, possibly for handling models.
- `templates/` - HTML templates for the project.
- `trainedModel/` - Directory for storing trained ML models.

## Files
- `.gitattributes` - Git configuration file.
- `README.md` - Project documentation.
- `RIL.csv` - Dataset file in CSV format.
- `TriningModel.ipynb` - Jupyter Notebook for training the model (**possible typo: "Trining" should be "Training"**).
- `db.sqlite3` - SQLite database file.
- `manage.py` - Django management script.
- `requirements.txt` - List of dependencies for the project.

## Notes
- Ensure that `requirements.txt` contains all necessary packages.
- Consider fixing the spelling of `TriningModel.ipynb` to `TrainingModel.ipynb`.
  
---

## Contributor
- **Shivanna** (shivannadm16@gmail.com)

---

## Acknowledgments
Special thanks to the open-source libraries and frameworks that made this project possible, including TensorFlow, Keras, and Scikit-learn.

---

## Contact
For inquiries or collaboration opportunities, feel free to reach out:
- **Email**: shivannadm6@gmail.com
- **LinkedIn**: [Shivanna DM](https://www.linkedin.com/in/shivannadm)

---

**Let's forecast the market with precision and agility!**
