from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from pydantic import BaseModel, validator

from datetime import datetime, timedelta
from jose import JWTError, jwt
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv
from typing import Dict
import os

from groq import Groq

client = Groq(
    api_key="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",
)

def get_response(promts):
    
    messages = [
        {
            "role": "system",
            "content": "You are professional therapist, please be patient and help people. Your name is Sentience. No need to introduce yourself if customer wont ask. Answer the questions as a therapist."
        }
    ]    
    
    for prompt in promts:
        messages.append({
            "role": "user",
            "content": prompt
        })
    
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama-3.3-70b-versatile", 
    )
       
    return chat_completion.choices[0].message.content

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "./saved_therapist_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.to(device)

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key")  # Fallback key for development
DATABASE_URL = os.getenv("DATABASE_URL", 'postgresql://sentience app_owner:npg_sLSq6d5xloJB@ep-small-bird-a5datc6e-pooler.us-east-2.aws.neon.tech/sentience app?sslmode=require')
GROK_API_KEY = os.getenv("GROK_API_KEY", "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")

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
    

class EstimateTestRequest(BaseModel):
    answers: Dict[str, int]
    @validator('answers')
    def convert_keys_to_int(cls, v):
        return {int(key): value for key, value in v.items()}

# def get_response(prompt, temperature=0.7, top_p=0.9, max_length=150):
    
#     if prompt in greeting:
#         return "Hello, I am Sentience. How you feeling?"
    
    
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#     output = model.generate(
#         **inputs,
#         max_length=max_length,
#         temperature=temperature,
#         top_p=top_p,
#         do_sample=True,
#         pad_token_id=tokenizer.eos_token_id
#     )
    
#     full_output = tokenizer.decode(output[0], skip_special_tokens=True)

#     if full_output.startswith(prompt):
#         return full_output[len(prompt):].strip()
#     return full_output.strip()


# Secret key for JWT
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800 
)

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
        
app = FastAPI()

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
            columns = ["id", "title", "body", "created_at", "author"]
            return {
                "articles": [
                    dict(zip(columns, article)) 
                    for article in articles
                ]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/articles/{article_id}")
def get_article(article_id: int, user: dict = Depends(get_current_user)):
    try:
        with engine.connect() as conn:
            articles = conn.execute(text("SELECT * FROM articles WHERE id = :id"), {"id": article_id}).fetchone()
            if not article:
                raise HTTPException(status_code=404, detail="Article not found")
            columns = ["id", "title", "body", "created_at", "author"]

            return {
                "articles": [
                    dict(zip(columns, article)) 
                    for article in articles
                ]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tests")
def get_tests(user: dict = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        tests = db.execute(text("SELECT * FROM tests")).fetchall()
        columns = ['id', 'title', 'description', 'created_at']
        print(tests[0])
        return {
                "tests": [
                    dict(zip(columns, test)) 
                    for test in tests
                ]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tests/{test_id}")
def get_test(test_id: int, user: dict = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
           # Получаем информацию о тесте
        test = db.execute(text("SELECT * FROM tests WHERE id = :id"), {"id": test_id}).fetchone()
        if not test:
            raise HTTPException(status_code=404, detail="Test not found")
        
        # Получаем вопросы для этого теста
        questions = db.execute(text("SELECT * FROM questions WHERE test_id = :test_id ORDER BY order_number"), {"test_id": test_id}).fetchall()
        
        # Получаем варианты ответов для каждого вопроса
        questions_with_options = []
        for question in questions:
            options = db.execute(text("SELECT * FROM answers WHERE question_id = :question_id"), {"question_id": question[0]}).fetchall()  # Используем индекс 0 для обращения к ID вопроса
            questions_with_options.append({
                "question": {
                    "id": question[0],
                    "test_id": question[1],
                    "text": question[2],
                    "order_number": question[3]
                },
                "options": [
                    {"id": option[0], "question_id": option[1], "option_text": option[2]}
                    for option in options
                ]
            })
        
        return {"test": {
            "id": test[0],
            "title": test[1],
            "description": test[2]
        }, "questions": questions_with_options}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.get("/tests_results")
def get_tests_results(user: dict = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        username = user["user_id"]
        db_user=  db.execute(text("SELECT * FROM users WHERE username = :username"), {"username": username}).fetchone()
        user_id = db_user[0]
        results = db.execute(text("SELECT * FROM user_results WHERE user_id = :user_id"), {"user_id": user_id}).fetchall()
        cols = ['id', 'user_id', 'test_id', 'total_score','result_text', 'created_at']
        
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
def estimate_test(
    test_id: int,
    request: EstimateTestRequest,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Преобразованные данные уже находятся в request.answers
        answers = request.answers
        print(f"Test ID: {test_id}, Answers: {answers}")
        
        username= user["user_id"]
        
        user = db.execute(text("SELECT * FROM users WHERE username = :username"), {"username": username}).fetchone()
        
        user_id = user[0] 
        
        # Получаем тест по ID
        test = db.execute(text("SELECT * FROM tests WHERE id = :id"), {"id": test_id}).fetchone()
        if not test:
            raise HTTPException(status_code=404, detail="Test not found")
        
        # Получаем вопросы для этого теста
        questions = db.execute(
            text("SELECT * FROM questions WHERE test_id = :test_id ORDER BY order_number"), 
            {"test_id": test_id}
        ).fetchall()

        # Подсчитываем правильные ответы
        score = 0
        for question in questions:

            question_id = question[0] 
            
            options = db.execute(
                text("SELECT * FROM answers WHERE question_id = :question_id"), 
                {"question_id": question_id}
            ).fetchall()
            
            print(f"Question ID: {question_id}, Options: {options}")
            print(f"Answers: {answers[question_id]}")
            for option in options:
                if option[0] == answers[question_id]:
                    score += option[3]*10

        
        
        print(f"Score: {score}")
        result_text = ""
        if score >=90 :
                result_text = "Results says that you need consulting with therapist"
        elif score >= 70:
                result_text = "Looks like everything is ok, but you need to be more careful"
        elif score >= 50:
                result_text = "You are fine"
        elif score >= 30:
                result_text = "Very small chance that you have problems"
        elif score >= 10:
                result_text = "Very small chance that you have problems"

                     
        # Сохраняем результаты теста в таблицу tests_results
        result = {
            "test_id": test_id,
            "user_id": user_id,
            "score": score,
            "created_at": datetime.utcnow(),
            "result_text": result_text
        }
        
        db.execute(text("""
            INSERT INTO user_results (test_id, user_id, total_score, result_text, created_at)
            VALUES (:test_id, :user_id, :total_score,:result_text, :created_at)
        """), {"test_id": test_id, "user_id": user_id, "total_score": result["score"], "result_text": result_text,  "created_at": result["created_at"]})
        db.commit()
        
        return {"message": "Test estimated successfully", "result": result}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/ask-ai")
def ask_ai(request: PromptRequest, user: dict = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        username = user["user_id"]
        prompt = request.prompt
      
        with engine.connect() as conn:
            user = db.execute(text("SELECT * FROM users WHERE username = :username"), {"username": username}).fetchone()
        
            user_id = user[0] 
           
            context = db.execute(text("SELECT * FROM messages WHERE user_id = :user_id"), {"user_id": user_id}).fetchall()
            prompts = []
            for message in context:
                prompts.append(message[2])
            prompts.append(prompt)    
            response = get_response(prompts)     
            db.execute(text("INSERT INTO messages (user_id, content, created_at) VALUES (:user_id, :content, CURRENT_TIMESTAMP)"),
                     {"user_id": user_id, "content": prompt})
            db.commit()
            print(response)       
            return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 