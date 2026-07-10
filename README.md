# AutoML-Nexus

# AutoML-Nexus 🚀

**AutoML-Nexus** is a Model Integrity & Pipeline Audit Engine. 

Think of this as a "controlled flight" for your machine learning projects. Instead of guessing how your data will behave or which model will work best, AutoML-Nexus automatically runs multiple experiments, audits the results, and gives you a clear report on what worked and why.

---

## 🔍 The "Why"
In many ML projects, the process is a "black box." You feed in data, you get a model, but you don't really know how it arrived there. **AutoML-Nexus** changes that by:
* **Automating the grunt work:** Detecting your problem type and picking the best algorithms.
* **Providing transparency:** Showing you the audit trail of every trial and optimization step.
* **Ensuring reliability:** Validating performance with rigorous metric checking before you trust the result.

---

## 🏗 How the System Works
We keep things organized so you can focus on the results rather than the clutter:

* **The Front Door (`/frontend`):** A modern, responsive dashboard where you upload files and track progress. It’s built to keep you informed, not overwhelmed.
* **The Engine (`/backend`):** This is where the heavy lifting happens. It is divided into specialized "departments":
    * **`automl/`**: The brain that detects if you need a classifier or a regressor.
    * **`optimization/`**: The tuner that fine-tunes your models to get the best accuracy.
    * **`evaluation/`**: The auditor that reviews performance metrics like a peer-reviewer.
    * **`models/`**: The vault where your successful models are safely stored.

---



## 📂 Project Structure

### Frontend (`/frontend`)
* `/src`: Contains the primary React application logic, state management, and UI components (`App.jsx`).
* `/public`: Houses static assets including logos, background images, and favicons.
* `/node_modules`: Dependencies required for the React environment.

### Backend (`/backend`)
* `/automl`: Orchestrates the ML lifecycle (`detector.py`, `model_selector.py`, `pipeline.py`).
* `/data`: Dataset ingestion, preprocessing, and train/test splitting workflows.
* `/evaluation`: Auditing tools including `metrics.py` and `validator.py`.
* `/models`: Persisted model architectures (`ann_model.py`, `ml_models.py`).
* `/optimization`: Optuna-based hyperparameter tuning engine and search-space definitions.
* `/utils`: Shared logging and system helper functions.
* `main.py`: FastAPI entry point coordinating API requests.
---

## 🛠 Tech Stack

* **Frontend:** React.js, Vite, Tailwind CSS, Lucide React (Icons), Axios.
* **Backend:** Python, FastAPI/Flask, Scikit-Learn, Pandas.

## ⚙️ Installation & Setup

### Prerequisites
* **Node.js**: v18 or higher
* **Python**: 3.9 or higher
* **npm** or **yarn** package manager

## 🛠 Getting Started
You don't need a PhD in DevOps to get this running. Just follow these steps:

### Frontend Steps
1. **Navigate to folder:** `cd frontend`
2. **Install dependencies:** `npm install`
3. **Launch:** `npm run dev`

### Backend Steps
1. **Navigate to folder:** `cd backend`
2. **Install requirements:** `pip install -r requirements.txt`
3. **Start API:** `uvicorn main:app --reload` (or equivalent start command)

---

## 📋 Usage Instructions

1. **Upload Data:** Select your dataset via the CSV file input in the dashboard.
2. **Select Target:** Use the dynamically generated dropdown to choose the specific target column for your machine learning model.
3. **Execute:** Click **"Run ML Pipeline"** to trigger the backend training process. Monitor live optimization trials and estimated completion times.
4. **Audit:** Once the pipeline completes, review the **System Validation Summary** and the **Run Manifest** to ensure model reliability.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

*Built with precision for professional ML auditing.*
