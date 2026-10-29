# Hosting RailScan with an external database

## What needs an external database, and what does not

| Data | Where it lives | Why |
|---|---|---|
| Railway data: 1,500 observations, 8,965 stations, timetable | SQLite (`data/database/railscan.db`) | Read-only. It is rebuilt automatically from the committed CSV files the first time the app starts on a new server, which takes a few seconds. |
| Accounts and history: `users`, `user_history`, `login_attempts` | **External PostgreSQL** when `RAILSCAN_ACCOUNTS_DB_URL` is set; otherwise the same SQLite file | Must survive restarts. Most hosts (including Streamlit Community Cloud) wipe local files when the app restarts, so accounts kept in SQLite there would be lost. |

The account code (`src/database/users.py` via `accounts_store.py`) runs unchanged on SQLite and PostgreSQL. `tests/test_users.py` runs every account test on both.

## Requirements of the host
- **Python 3.11** and `pip install -r requirements.txt`. The file pins CPU-only PyTorch, so no GPU packages are downloaded.
- **About 2 GB of RAM or more.** Measured locally, the app peaks at about 1.8 GB during a sign scan, when TensorFlow, PyTorch and the EasyOCR models are loaded. The journey planner, history and log-in pages use far less. A host with less memory will restart the app on the first scan.
- About 3.5 GB of disk for the installed packages, plus about 330 MB for the EasyOCR models. The models download on the first scan of each script, and again after every restart on hosts without persistent disk.

---

## Step 1 - Create the PostgreSQL database
Any PostgreSQL 13 or newer works. Two providers with free tiers (check their current terms):

**Neon** (https://neon.tech)
1. Sign up and choose **Create project**. Pick a region near your users (for example AWS Asia Pacific, Mumbai or Singapore).
2. On the project dashboard, open **Connection details** and copy the connection string. It looks like:
   `postgresql://railscan_owner:YOUR_PASSWORD@ep-example-123456.ap-southeast-1.aws.neon.tech/neondb?sslmode=require`

**Supabase** (https://supabase.com)
1. Create a project and set a database password.
2. Choose **Connect** at the top of the project page. Under **Connection string → URI**, select the **Session pooler** method. It looks like:
   `postgresql://postgres.abcdefghijkl:[YOUR-PASSWORD]@aws-0-ap-south-1.pooler.supabase.com:5432/postgres`
3. Put your password in place of `[YOUR-PASSWORD]` (without the brackets) and add `?sslmode=require` at the end.
   - Use the session pooler rather than the **Direct connection** (`db.<ref>.supabase.co`). On the free plan the direct connection only works over IPv6, which many home networks and hosts, including Streamlit Community Cloud, do not have.
   - The **Transaction pooler** (port 6543) also works. RailScan turns off prepared statements for it.

You do **not** need to create any tables. RailScan creates `users`, `user_history` and `login_attempts` (with indexes) on its first connection.

**Security notes**
- Keep the connection string secret: it contains the password. Never commit it to git. `.gitignore` already excludes `.streamlit/secrets.toml` and `.env`.
- Keep `sslmode=require`, so the connection is encrypted.
- Passwords are stored only as salted scrypt hashes. History contains the results users saw (station names, delays, OCR text), never photos. Deleting an account deletes its history.
- If you can, create a separate database user that can only use this database, and turn on the provider's backups.

## Step 2 - Test it from your PC first
In PowerShell, in the project folder:
```powershell
.\.venv\Scripts\Activate.ps1
$env:RAILSCAN_ACCOUNTS_DB_URL = "postgresql://...your connection string..."
streamlit run app.py --server.port 8502
```
1. Open http://localhost:8502, create an account and run a planner estimate.
2. Open **My history**. The account line should end with "stored in PostgreSQL at <your host>".
3. In the provider's SQL editor, `SELECT username, created_at FROM users;` shows the account. The password column holds only a `scrypt$...` hash.

To go back to local SQLite accounts, close the terminal (the variable only lasts for that window) or run `Remove-Item Env:RAILSCAN_ACCOUNTS_DB_URL`.

## Step 3 - Put the code on GitHub
```powershell
git init
git add .
git commit -m "RailScan"
git branch -M main
git remote add origin https://github.com/<your-account>/railscan.git
git push -u origin main
```

**What gets committed:** code, processed CSVs (about 2 MB), the delay models (about 40 KB) and the CNN model (about 24 MB). All are below GitHub's 100 MB file limit, so Git LFS is not needed.

**Not committed:** `.venv/`, `data/raw/`, `data/sign_images/`, `data/interim/`, the SQLite file and secrets. Before pushing, check with `git status` that no `.db` file and no `secrets.toml` is listed.

## Step 4a - Host on Streamlit Community Cloud
1. Go to https://share.streamlit.io, sign in with GitHub and choose **Create app**.
2. Select the repository, branch `main` and main file `app.py`.
3. Open **Advanced settings**. Choose Python **3.11**, and paste this into **Secrets**:
   ```toml
   RAILSCAN_ACCOUNTS_DB_URL = "postgresql://...your connection string..."
   ```
4. Choose **Deploy**. The first build installs `requirements.txt`, which takes several minutes because of TensorFlow and PyTorch.
5. On the first visit, the app builds the railway tables ("Preparing the railway database"). The first scan of each script downloads EasyOCR's models.

Check the memory limit of the free tier in Streamlit's current documentation against the measured peak of about 1.8 GB. If the app restarts during a scan, use a host with more memory (step 4b). Everything except scanning, including log-in, the planner and history, uses much less memory.

## Step 4b - Any host that runs Python (Render, Railway, a VM, ...)
Use these settings:
- **Build command:** `pip install -r requirements.txt`
- **Start command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
  - On a VM, use a fixed port instead of `$PORT`, for example 8501, and put it behind HTTPS (for example Nginx or Caddy).
- **Environment variable:** `RAILSCAN_ACCOUNTS_DB_URL` = your connection string, entered in the host's secret or environment settings.
- **Python version:** 3.11.
- **Memory:** 2 GB or more.

## Step 5 - Check the deployment
- Create an account, run a planner estimate, and open **My history**. It should show the entry and "stored in PostgreSQL at …".
- Restart or redeploy the app and log in again. The account and history must still be there; if they are not, the secret was not picked up.
- **Data and models** should say the railway data is read from the SQLite database.

## Troubleshooting
| Symptom | Cause / fix |
|---|---|
| Accounts disappear after a restart | `RAILSCAN_ACCOUNTS_DB_URL` is not set on the host, so accounts went to the temporary SQLite file. Check the secret's exact name. |
| `connection refused` / `timeout` on sign-up | The connection string's host or port is wrong, or the provider blocks the connection. Check that `sslmode=require` is present and the database is not paused (free tiers pause after inactivity and wake on the next connection). |
| `password authentication failed` | The password in the URL is wrong, or contains characters that must be URL-encoded (`@` → `%40`, `:` → `%3A`, `/` → `%2F`). |
| App restarts while scanning | Not enough memory (needs about 2 GB). |
| First scan very slow | EasyOCR is downloading its models; later scans are faster. |
| "PostgreSQL accounts need psycopg" | `psycopg[binary]` is missing; reinstall from `requirements.txt`. |

## What was tested
- **Tested locally on a real PostgreSQL 16 server** (the `pgserver` package):
  - all account tests (sign-up, hashing, unique usernames, lockout, private history, password change, account deletion);
  - a full app run with `RAILSCAN_ACCOUNTS_DB_URL` set: sign-up, a planner estimate saved to history, and the rows checked in PostgreSQL;
  - the automatic railway-database build on an empty server.
- **Not tested:** a real Neon or Supabase database, and deployment to Streamlit Community Cloud or another host. No accounts on those services were used in this work.
