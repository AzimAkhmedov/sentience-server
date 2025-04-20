
# 📘 Sentience API

API для взаимодействия с AI-терапевтом, прохождения тестов, чтения статей и авторизации пользователей.

## 🔐 Аутентификация

Все защищённые маршруты требуют **JWT токен** в заголовке:

```
Authorization: Bearer <your_token>
```

---

## 🔑 Auth

### ▶️ POST `/register`
Зарегистрировать нового пользователя.

**Request:**
```json
{
  "username": "johndoe",
  "password": "123456",
  "profile_pic": "https://example.com/avatar.png"
}
```

**Response:**
```json
{
  "message": "User registered successfully",
  "access_token": "<jwt_token>"
}
```

---

### ▶️ POST `/login`
Авторизация пользователя.

**Request:**
```json
{
  "username": "johndoe",
  "password": "123456"
}
```

**Response:**
```json
{
  "access_token": "<jwt_token>",
  "token_type": "bearer"
}
```

---

### 🔐 GET `/me`
Получить данные о текущем пользователе.

**Headers:**
```
Authorization: Bearer <jwt_token>
```

**Response:**
```json
{
  "message": "This is a protected route",
  "user": {
    "user_id": "johndoe"
  }
}
```

---

## 📝 Articles

### 🔐 GET `/articles`
Получить список всех статей.

### 🔐 GET `/articles/{article_id}`
Получить одну статью по ID.

---

## 🧠 Tests

### 🔐 GET `/tests`
Получить список всех доступных тестов.

### 🔐 GET `/tests/{test_id}`
Получить данные по одному тесту (и его опциям).

---

## 📊 Test Results

### 🔐 GET `/tests_results`
Получить все результаты тестов текущего пользователя.

### 🔐 GET `/tests_results/{result_id}`
Получить один результат теста по ID.

---

### ▶️ POST `/estimate-test/{test_id}`
Оценить тест и сохранить результат.

**Headers:**
```
Authorization: Bearer <jwt_token>
```

**Request:**
```json
{
  "1": "A",
  "2": "C",
  "3": "B"
}
```

**Response:**
```json
{
  "message": "Test estimated successfully",
  "result": {
    "test_id": 1,
    "user_id": "johndoe",
    "score": 85,
    "created_at": "2025-04-19T12:00:00Z"
  }
}
```

---

## 🧠 Ask AI

### ▶️ POST `/ask-ai`
Отправить промпт AI-терапевту и получить ответ.

**Headers:**
```
Authorization: Bearer <jwt_token>
```

**Request:**
```json
{
  "prompt": "Я чувствую тревогу и стресс"
}
```

**Response:**
```json
{
  "response": "Попробуйте сделать глубокий вдох и поговорите с близким человеком о своих переживаниях."
}
```

---

## 🛡️ Security

Используется `JWT` в формате Bearer-токена:
```http
Authorization: Bearer <jwt_token>
```
