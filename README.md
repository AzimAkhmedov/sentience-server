# Sentience Server

This repository contains all backend functionality to implement ai-therapist.

## Setting up

Install required dependencies from the list

```bash
  pip install -r dependencies.txt
```

Configure .env.local

```bash
  SECRET_KEY="your_secret_key"
  DB_URL="postgresql://username:password@host/dbname?sslmode=require"
```

Run localy

```bash
  python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Now you can easly check your 8000 port
