# Ed Tech CodePro Lead Classification MLOps Project

Welcome to the Ed Tech CodePro Lead Classification project—a state-of-the-art MLOps assignment designed to streamline the end-to-end process of data ingestion, cleaning, training, and inference for lead scoring. This repository houses modular pipelines, interactive notebooks, and robust unit tests that combine to deliver a scalable and maintainable solution for lead classification.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Airflow Setup Instructions](#airflow-setup-instructions)
- [Pipeline Overview](#pipeline-overview)
  - [Data Pipeline](#data-pipeline)
  - [Training Pipeline](#training-pipeline)
  - [Inference Pipeline](#inference-pipeline)
  - [Master Pipeline DAG](#master-pipeline-dag)
  - [Unit Testing](#unit-testing)
- [Notebooks](#notebooks)
- [Running the Project](#running-the-project)
- [Contributing](#contributing)
- [License](#license)
- [Contact & Acknowledgements](#contact--acknowledgements)

---

## Project Overview

This project implements a full-fledged MLOps pipeline to classify leads for Ed Tech using a modular architecture orchestrated by Apache Airflow. It integrates:

- **Data Ingestion & Cleaning:** Automated scripts to validate, clean, and prepare raw lead data.
- **Model Training:** A dedicated training pipeline that experiments with and refines models.
- **Inference & Prediction:** A pipeline for real-time prediction and lead scoring.
- **Testing & Validation:** Robust unit tests ensuring the integrity of each stage.

---

## Features

- **Automated Data Processing:** Clean and validate your data with a dedicated pipeline.
- **Model Training & Inference:** Seamless pipelines for building and deploying machine learning models.
- **Airflow-Orchestrated Workflows:** Manage complex dependencies and schedules with ease.
- **Interactive Analysis:** Leverage Jupyter notebooks for exploratory data analysis and model experimentation.
- **Modular & Scalable:** Easily extend or modify each component of the system.

---

## Repository Structure

```
airflow
├── airflow.cfg
├── dags
│   ├── __init__.py
│   ├── lead_scoring_data_pipeline
│   │   ├── __init__.py
│   │   ├── constants.py
│   │   ├── data
│   │   │   ├── leadscoring.csv
│   │   │   └── leadscoring_inference.csv
│   │   ├── data_validation_checks.py
│   │   ├── lead_scoring_data_cleaning.db
│   │   ├── lead_scoring_data_pipeline.py
│   │   ├── mappings
│   │   │   ├── city_tier_mapping.py
│   │   │   ├── interaction_mapping.csv
│   │   │   └── significant_categorical_level.py
│   │   ├── schema.py
│   │   └── utils.py
│   ├── lead_scoring_inference_pipeline
│   │   ├── __init__.py
│   │   ├── constants.py
│   │   ├── lead_scoring_inference_pipeline.py
│   │   ├── prediction_distribution.txt
│   │   ├── schema.py
│   │   └── utils.py
│   ├── lead_scoring_training_pipeline
│   │   ├── __init__.py
│   │   ├── constants.py
│   │   ├── lead_scoring_training_pipeline.py
│   │   └── utils.py
│   ├── master_pipeline_dag.py
│   └── unit_test
│       ├── __init__.py
│       ├── constants.py
│       ├── leadscoring_test.csv
│       ├── test_runner_dag.py
│       ├── test_with_pytest.py
│       └── unit_test_cases.db
notebooks
├── 01.data_cleaning.ipynb
├── 02.model_experimentation.ipynb
├── data
│   ├── cleaned_data.csv
│   └── leadscoring.csv
├── mappings
│   ├── city_tier_mapping.py
│   ├── interaction_mapping.csv
│   └── significant_categorical_level.py
└── profile_reports
    ├── cleaned_data_report.html
    └── raw_data_report.html
screenshots.pdf
webserver_config.py
```

**Folder Highlights:**

- **airflow:** Contains Airflow configuration and DAGs for each pipeline component, including data cleaning, model training, inference, and unit testing.
- **notebooks:** Jupyter notebooks and supplementary files for exploratory data analysis and model experimentation.
- **screenshots.pdf:** Visual documentation of key pipeline outputs and dashboards.
- **webserver_config.py:** Configuration file for setting up the Airflow web server.

---

## Getting Started

### Prerequisites

- **Python:** Version 3.7 or higher
- **Apache Airflow:** Installed and configured (see instructions below)
- **Jupyter Notebook:** For running and modifying interactive notebooks
- **Other Dependencies:** Listed in `requirements.txt` (if available)

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. **Set Up a Virtual Environment & Install Dependencies:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure Apache Airflow:**  
   Ensure you have a proper Airflow setup as described in the [Airflow Setup Instructions](#airflow-setup-instructions).

---

## Airflow Setup Instructions

Follow these steps to set up Airflow:

### A. Update `airflow.cfg`

1. Open the `airflow.cfg` file.
2. Make the following changes:
   ```
   base_url = http://localhost:6007
   ```
   ```
   web_server_port = 6007
   ```

---

### B. Run Initialization Commands

1. **Initialize the Airflow metadata database:**

   ```bash
   airflow db init
   ```

2. **Create an admin user to access the Airflow UI:**

   ```bash
   airflow users create \
       --username upgrad \
       --firstname upgrad \
       --lastname upgrad \
       --role Admin \
       --email spiderman@superhero.org \
       --password admin
   ```

---

### C. Start Airflow

1. **Start the Airflow web server:**

   ```bash
   airflow webserver
   ```

2. **In a new terminal, start the scheduler:**

   ```bash
   airflow scheduler
   ```

---

### Clean Restart (Optional)

If you want to completely reset your Airflow setup and start fresh:

1. **Reset the database (this will erase all metadata):**

   ```bash
   airflow db reset --yes
   ```

2. **Re-initialize the metadata database:**

   ```bash
   airflow db init
   ```

3. **Recreate the admin user:**

   ```bash
   airflow users create \
       --username upgrad \
       --firstname upgrad \
       --lastname upgrad \
       --role Admin \
       --email spiderman@superhero.org \
       --password admin
   ```

---

## Pipeline Overview

### Data Pipeline

- **Location:** `airflow/dags/lead_scoring_data_pipeline/`
- **Purpose:** Ingest raw lead data, perform data cleaning, run validation checks, and prepare data for downstream processes.
- **Key Components:**
  - Data validation scripts and cleaning utilities.
  - Mapping modules (e.g., city tier, interaction mapping) to enrich raw data.

### Training Pipeline

- **Location:** `airflow/dags/lead_scoring_training_pipeline/`
- **Purpose:** Train machine learning models for lead classification using processed data.
- **Key Components:**
  - Model training scripts.
  - Constants and utility functions to manage training configurations.

### Inference Pipeline

- **Location:** `airflow/dags/lead_scoring_inference_pipeline/`
- **Purpose:** Generate predictions using the trained model.
- **Key Components:**
  - Prediction scripts.
  - Schema definitions to ensure prediction consistency.

### Master Pipeline DAG

- **Location:** `airflow/dags/master_pipeline_dag.py`
- **Purpose:** Orchestrate the overall workflow by coordinating the data, training, and inference pipelines.
- **Key Components:**
  - Workflow scheduling and dependency management.

### Unit Testing

- **Location:** `airflow/dags/unit_test/`
- **Purpose:** Validate pipeline functionality and data integrity through automated tests.
- **Key Components:**
  - Pytest-based test scripts and test data.
  - A dedicated test runner DAG for executing tests within Airflow.

---

## Notebooks

The `notebooks` directory provides interactive resources for:
- **Data Cleaning:** (`01.data_cleaning.ipynb`) – Explore, visualize, and clean raw data.
- **Model Experimentation:** (`02.model_experimentation.ipynb`) – Experiment with different models and parameters.
- **Supplementary Files:**  
  - **Data Files:** Raw and cleaned datasets.
  - **Mapping Scripts:** For data transformation.
  - **Profile Reports:** Detailed HTML reports summarizing data quality.

---

## Running the Project

1. **Start Airflow:**  
   Follow the [Airflow Setup Instructions](#airflow-setup-instructions) to initialize and run the Airflow webserver and scheduler.

2. **Trigger DAGs:**  
   From the Airflow UI, trigger the master pipeline DAG to run the complete end-to-end workflow—from data cleaning to model inference.

3. **Explore Notebooks:**  
   Open the notebooks in Jupyter to interactively analyze data and experiment with models.

4. **Run Tests:**  
   Execute unit tests to validate pipeline integrity:
   ```bash
   pytest dags/unit_test/test_runner_dag.py
   ```

---

## License

This project is open-source and available under the [MIT License](LICENSE).

---

## Contact & Acknowledgements

For questions or further information, please reach out to:

- **Maintainer:** [Amit Mohite]
- **Email:** [mohite.amit@gmail.com]
