from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import JWTError, jwt
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

# Access the environment variables
SECRET_KEY = os.getenv("SECRET_KEY")
DATABASE_URL = 'postgresql://sentience app_owner:npg_sLSq6d5xloJB@ep-small-bird-a5datc6e-pooler.us-east-2.aws.neon.tech/sentience app?sslmode=require'


# Secret key for JWT
SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Database URL
engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(bind=engine)

# FastAPI app
app = FastAPI()

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def hash_password(password):
    return pwd_context.hash(password)

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"user_id": user_id}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/register")
def register_user(username: str, password: str, profile_pic: str, ):
    hashed_password = hash_password(password)
    try:
        with engine.connect() as conn:
            conn.execute(text("INSERT INTO users (username, password, profile_img, created_at, role) VALUES (:username, :password, :profile_img, CURRENT_TIMESTAMP, 'user')"),
                         {"username": username, "password": hashed_password, "profile_img": profile_pic})
            conn.commit()
            access_token = create_access_token({"sub": username})
            
            return {"message": "User registered successfully", "access_token": access_token}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    try:
        with engine.connect() as conn:
            user = conn.execute(text("SELECT * FROM users WHERE username = :username"),
                                {"username": form_data.username}).fetchone()
            if not user or not verify_password(form_data.password, user.password):
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
            return {"articles": [dict(article) for article in articles]}
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
            tests = conn.execute(text("SELECT * FROM tests")).fetchall()
            return {"tests": [dict(test) for test in tests]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tests/{test_id}") 
def get_test(test_id: int, user: dict = Depends(get_current_user)):
    try:
        with engine.connect() as conn:
            test = conn.execute(text("SELECT * FROM test_options WHERE id = :id"), {"id": test_id}).fetchone()
            if not test:
                raise HTTPException(status_code=404, detail="Test not found")
            return {"test": dict(test)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tests_results")
def get_tests_results(user: dict = Depends(get_current_user)):
    try:
        user_id = user["user_id"]
        with engine.connect() as conn:
            results = conn.execute(text("SELECT * FROM tests_results WHERE user_id = :user_id"), {"user_id": user_id}).fetchall()
            return {"results": [dict(result) for result in results]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tests_results/{result_id}")
def get_test_result(result_id: int, user: dict = Depends(get_current_user)):
    try:
        user_id = user["user_id"]
        with engine.connect() as conn:
            result = conn.execute(text("SELECT * FROM tests_results WHERE id = :id AND user_id = :user_id"),
                                  {"id": result_id, "user_id": user_id}).fetchone()
            if not result:
                raise HTTPException(status_code=404, detail="Test result not found")
            return {"result": dict(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
    
@app.post("/estimate-test/{test_id}")
def estimate_test(test_id: int,answers: dict, user: dict = Depends(get_current_user)):
    try:
        user_id = user["user_id"]
        with engine.connect() as conn:
            test = conn.execute(text("SELECT * FROM test_options WHERE id = :id"), {"id": test_id}).fetchone()
            test_options = conn.execute(text("SELECT * FROM test_options WHERE test_id = :test_id"), {"test_id": test_id}).fetchall()
            if not test:
                raise HTTPException(status_code=404, detail="Test not found")
            
            if not test_options:
                raise HTTPException(status_code=404, detail="Test options not found")            
             
            result = {
                "test_id": test_id,
                "user_id": user_id,
                "score": 85,  # TODO: calculate score
                "created_at": datetime.utcnow()
            }
            
            conn.execute(text("INSERT INTO tests_results (test_id, user_id, score, created_at) VALUES (:test_id, :user_id, :score, :created_at)"),
                         {"test_id": test_id, "user_id": user_id, "score": result["score"], "created_at": result["created_at"]})
            conn.commit()
            
            return {"message": "Test estimated successfully", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))