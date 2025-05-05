from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import JWTError, jwt
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv
from typing import Dict
import os


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Auto-detect device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer and model
model_path = "./saved_therapist_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.to(device)

# Load the .env file
load_dotenv()

# Access the environment variables
SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key")  # Fallback key for development
DATABASE_URL = os.getenv("DATABASE_URL", 'postgresql://sentience app_owner:npg_sLSq6d5xloJB@ep-small-bird-a5datc6e-pooler.us-east-2.aws.neon.tech/sentience app?sslmode=require')

from pydantic import BaseModel


class PromptRequest(BaseModel):
    prompt: str

class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str
    profile_pic: str

def get_response(prompt, temperature=0.7, top_p=0.9, max_length=150):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    full_output = tokenizer.decode(output[0], skip_special_tokens=True)

    if full_output.startswith(prompt):
        return full_output[len(prompt):].strip()
    return full_output.strip()

# Secret key for JWT
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Create database engine with connection pooling and better error handling
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Enable connection health checks
    pool_size=5,  # Set a reasonable pool size
    max_overflow=10,  # Allow some overflow connections
    pool_timeout=30,  # Timeout for getting a connection from the pool
    pool_recycle=1800  # Recycle connections after 30 minutes
)

# Create tables if they don't exist
with engine.connect() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS tests (
            id SERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS test_options (
            id SERIAL PRIMARY KEY,
            test_id INTEGER REFERENCES tests(id),
            option_text TEXT NOT NULL,
            score INTEGER NOT NULL
        )
    """))
    conn.commit()

SessionLocal = sessionmaker(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        
# FastAPI app
app = FastAPI()

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def hash_password(password):
    return pwd_context.hash(password)

def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # Strip "Bearer " prefix if present
        if token.startswith("Bearer "):
            token = token[7:]
            
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        return {"user_id": user_id}
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.JWTError:
        raise credentials_exception

@app.post("/register")
def register_user(request: RegisterRequest):
    hashed_password = hash_password(request.password)
    try:
        with engine.connect() as conn:
            conn.execute(text("INSERT INTO users (username, password, profile_img, created_at, role) VALUES (:username, :password, :profile_img, CURRENT_TIMESTAMP, 'user')"),
                         {"username": request.username, "password": hashed_password, "profile_img": request.profile_pic})
            conn.commit()
            access_token = create_access_token({"sub": request.username})
            
            return {"message": "User registered successfully", "access_token": access_token}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/login")
def login(request: LoginRequest):
    try:
        with engine.connect() as conn:
            user = conn.execute(text("SELECT * FROM users WHERE username = :username"),
                                {"username": request.username}).fetchone()
            if not user or not verify_password(request.password, user.password):
                raise HTTPException(status_code=400, detail="Invalid username or password")
            access_token = create_access_token({"sub": user.username})
            return {"access_token": access_token, "token_type": "bearer"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/me")
def private_route(user: dict = Depends(get_current_user)):
    return {"message": "This is a protected route", "user": user}

@app.get("/articles")
def get_articles(user: dict = Depends(get_current_user)):
    try:
        with engine.connect() as conn:
            articles = conn.execute(text("SELECT * FROM articles")).fetchall()
            # Convert each row to a dictionary and return as a list
            return [dict(row._mapping) for row in articles]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/articles/{article_id}")
def get_article(article_id: int, user: dict = Depends(get_current_user)):
    try:
        with engine.connect() as conn:
            article = conn.execute(text("SELECT * FROM articles WHERE id = :id"), {"id": article_id}).fetchone()
            if not article:
                raise HTTPException(status_code=404, detail="Article not found")
            return {"article": dict(article)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tests")
def get_tests(user: dict = Depends(get_current_user)):
    try:
        with engine.connect() as conn:
            tests = conn.execute(text("""
                SELECT t.*, json_agg(test_options.*) as options 
                FROM tests t
                LEFT JOIN test_options ON t.id = test_options.test_id
                GROUP BY t.id
            """)).fetchall()
            # Convert each row to a dictionary and return as a list
            return [dict(row._mapping) for row in tests]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tests/{test_id}")
def get_test(test_id: int, user: dict = Depends(get_current_user)):
    try:
        with engine.connect() as conn:
            test = conn.execute(text("""
                SELECT t.*, json_agg(test_options.*) as options 
                FROM tests t
                LEFT JOIN test_options ON t.id = test_options.test_id
                WHERE t.id = :test_id
                GROUP BY t.id
            """), {"test_id": test_id}).fetchone()
            
            if not test:
                raise HTTPException(status_code=404, detail="Test not found")
            
            return dict(test._mapping)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.get("/tests_results")
def get_tests_results(user: dict = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        username = user["user_id"]
        db_user=  db.execute(text("SELECT * FROM users WHERE username = :username"), {"username": username}).fetchone()
        print(db_user)
        user_id = db_user[0]
        print(user_id)
        results = db.execute(text("SELECT * FROM user_results WHERE user_id = :user_id"), {"user_id": user_id}).fetchall()
        cols = ['id', 'user_id', 'test_id', 'total_score','result_text', 'created_at']
        print(results)
        
                    # dict(zip(columns, test)) 
                    # for test in tests
        
        return {"results": [dict(zip(cols,result)) for result in results]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tests_results/{result_id}")
def get_test_result(result_id: int, user: dict = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        user_id = user["user_id"]
        result = db.execute(text("SELECT * FROM tests_results WHERE id = :id AND user_id = :user_id"),
                            {"id": result_id, "user_id": user_id}).fetchone()
        if not result:
            raise HTTPException(status_code=404, detail="Test result not found")
        return {"result": dict(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/estimate-test/{test_id}")
def estimate_test(test_id: int, answers: Dict[int, int], user: dict = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        user_id = user["user_id"]
        
        # Получаем тест по ID
        test = db.execute(text("SELECT * FROM tests WHERE id = :id"), {"id": test_id}).fetchone()
        if not test:
            raise HTTPException(status_code=404, detail="Test not found")
        
        # Получаем вопросы для этого теста
        questions = db.execute(text("SELECT * FROM questions WHERE test_id = :test_id ORDER BY order_number"), {"test_id": test_id}).fetchall()
        
        # Подсчитываем правильные ответы
        score = 0
        for question in questions:
            options = db.execute(text("SELECT * FROM test_options WHERE question_id = :question_id"), {"question_id": question["id"]}).fetchall()
            correct_option = next((opt for opt in options if opt["is_correct"] == True), None)
            
            # Проверяем, совпадает ли ответ с правильным
            if correct_option and answers.get(question["id"]) == correct_option["id"]:
                score += 1
        
        # Сохраняем результаты теста в таблицу tests_results
        result = {
            "test_id": test_id,
            "user_id": user_id,
            "score": score,
            "created_at": datetime.utcnow()
        }
        
        db.execute(text("""
            INSERT INTO tests_results (test_id, user_id, score, created_at)
            VALUES (:test_id, :user_id, :score, :created_at)
        """), {"test_id": test_id, "user_id": user_id, "score": result["score"], "created_at": result["created_at"]})
        db.commit()
        
        return {"message": "Test estimated successfully", "result": result}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask-ai")
def ask_ai(request: PromptRequest, user: dict = Depends(get_current_user)):
    try:
        user_id = user["user_id"]
        prompt = request.prompt
        with engine.connect() as conn:
            response = get_response(prompt)            
            return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 