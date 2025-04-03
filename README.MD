# Replace 'AUC' with the actual name of the metric you logged (e.g., 'auc', 'accuracy')
query = """
SELECT run_uuid, key, value
FROM metrics
WHERE key = 'AUC'
ORDER BY value DESC
LIMIT 1
"""

best_run_df = pd.read_sql(query, conn)
conn.close()

if best_run_df.empty:
    print("❌ No runs found with the specified metric.")
else:
    best_run_id = best_run_df.loc[0, 'run_uuid']
    best_auc = best_run_df.loc[0, 'value']
    print(f"✅ Best run ID: {best_run_id}, AUC: {best_auc}")

-------

import mlflow

mlflow.set_tracking_uri("http://0.0.0.0:6006")  # or wherever your MLflow server is

best_model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")

-------


pip install psutil
pip install optuna==2.10.0

## Airflow Setup Instructions

### A. Update `airflow.cfg`

1. Open the `airflow.cfg` file.
2. Make the following changes:
   - Update the `base_url`:
     ```
     base_url = http://localhost:6007
     ```
   - Update the web server port:
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
