# ✅ Summary: Google BigQuery Setup & Authentication

## 🔧 Project Setup

- **Project name**: ` `
- **Project ID**:   ` `  not just name but also ID numbers

## 🔐 Authentication

- Installed **Google Cloud SDK**
- Set active account email:  ` `
- Set project:

  ```bash
  gcloud config set project !!! Project_ID
  ```

- Authenticated using:

  ```bash
  gcloud auth application-default login
  ```

- Consent granted for required scopes (e.g. `cloud-platform`, `userinfo.email`, etc.)

- Credentials stored locally at:

  ```
  application_default_credentials.json
  ```

## 🧪 Status

- ✅ Application Default Credentials ready  
- ✅ BigQuery API enabled  
- ✅ Project config verified
