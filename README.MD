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
