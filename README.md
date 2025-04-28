
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

## 1. Получить все тесты

**URL**: `/tests`  
**Method**: `GET`

### Description:
Получить список всех доступных тестов.

### Returns:
- `tests`: Список всех тестов. Каждый тест включает поля:
  - `id`: ID теста.
  - `title`: Название теста.
  - `description`: Описание теста.

### Пример ответа:
```json
{
  "tests": [
    {
      "id": 1,
      "title": "Эмоциональное выгорание",
      "description": "Тест на определение уровня эмоционального выгорания."
    },
    {
      "id": 2,
      "title": "Самооценка",
      "description": "Тест на оценку уровня самооценки."
    }
  ]
}
```

---

## 2. Получить тест по ID

**URL**: `/tests/{test_id}`  
**Method**: `GET`

### Parameters:
- `test_id`: ID теста.

### Description:
Получить информацию о тесте по его ID.

### Returns:
- `test`: Детали теста.
  - `id`: ID теста.
  - `title`: Название теста.
  - `description`: Описание теста.

### Пример ответа:
```json
{
  "test": {
    "id": 1,
    "title": "Эмоциональное выгорание",
    "description": "Тест на определение уровня эмоционального выгорания."
  }
}
```

---

## 3. Получить результаты всех тестов пользователя

**URL**: `/tests_results`  
**Method**: `GET`

### Description:
Получить результаты всех тестов пользователя.

### Returns:
- `results`: Список результатов тестов пользователя.
  - `id`: ID результата.
  - `test_id`: ID теста.
  - `score`: Оценка, полученная пользователем.
  - `created_at`: Дата и время прохождения теста.

### Пример ответа:
```json
{
  "results": [
    {
      "id": 1,
      "test_id": 1,
      "score": 80,
      "created_at": "2025-04-28T12:30:00"
    },
    {
      "id": 2,
      "test_id": 2,
      "score": 90,
      "created_at": "2025-04-27T15:00:00"
    }
  ]
}
```

---

## 4. Получить результаты теста по ID

**URL**: `/tests_results/{result_id}`  
**Method**: `GET`

### Parameters:
- `result_id`: ID результата теста.

### Description:
Получить подробности результата теста по его ID.

### Returns:
- `result`: Детали результата теста.
  - `id`: ID результата.
  - `test_id`: ID теста.
  - `score`: Оценка, полученная пользователем.
  - `created_at`: Дата и время прохождения теста.

### Пример ответа:
```json
{
  "result": {
    "id": 1,
    "test_id": 1,
    "score": 80,
    "created_at": "2025-04-28T12:30:00"
  }
}
```

---

## 5. Оценить тест

**URL**: `/estimate-test/{test_id}`  
**Method**: `POST`

### Parameters:
- `test_id`: ID теста.
- `answers`: Ответы пользователя на вопросы теста (словарь с вопросами и ответами).
- `user`: Информация о пользователе (извлекается через `Depends(get_current_user)`).

### Description:
Оценить тест на основе ответов пользователя и сохранить результат в базе данных.

### Request Body:
```json
{
  "answers": {
    "1": 3,
    "2": 4,
    "3": 5
  }
}
```

### Returns:
- `message`: Сообщение об успешной оценке теста.
- `result`: Детали результата теста.
  - `test_id`: ID теста.
  - `user_id`: ID пользователя.
  - `score`: Оценка, полученная пользователем.
  - `created_at`: Дата и время прохождения теста.

### Пример ответа:
```json
{
  "message": "Test estimated successfully",
  "result": {
    "test_id": 1,
    "user_id": 123,
    "score": 85,
    "created_at": "2025-04-28T12:45:00"
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
