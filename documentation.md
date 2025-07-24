# ID Card API Documentation

## Overview
The **ID Card API** is a FastAPI-based service that processes images of ID cards and extracts key information. It provides secure authentication via API keys, rate limiting, and structured JSON responses.

---

## Base URL
```
http://yourserver.com:8000
```

---

## Authentication
The API uses **API Key authentication**. Clients must provide the following headers in each request:

- **`X-API-Key`**: API Key for authentication.
- **`X-API-Secret`**: API Secret for authentication.

### Example Request Headers:
```http
X-API-Key: your_api_key
X-API-Secret: your_api_secret
```

---

## Endpoints

### 1. Process ID Card
**`POST /api/id-recognition`**

Processes an uploaded ID card image and returns extracted information.

#### Request:
- **Headers:**
  - `X-API-Key`: (Required) API Key for authentication.
  - `X-API-Secret`: (Required) API Secret for authentication.

- **Form Data:**
  - `file`: (Required) The ID card image (JPG/PNG).

#### Response:
- **Status 200 (Success)**
  ```json
  {
    "timestamp": 1712345678.12,
    "accuracy": 0.95,
    "id_data": {
      "first_name": "John",
      "last_name": "Doe",
      "national_number": "1234567890",
      "address": "123 Main Street, City, Country"
    }
  }
  ```

- **Status 401 (Unauthorized)**
  ```json
  {
    "detail": "Invalid API credentials"
  }
  ```

- **Status 429 (Rate Limit Exceeded)**
  ```json
  {
    "detail": "Rate limit exceeded. Try again later."
  }
  ```

- **Status 500 (Internal Error)**
  ```json
  {
    "detail": "An unexpected error occurred"
  }
  ```

---

## Features

### 1. **Authentication & Security**
- Uses **API Key & Secret** for authentication.
- Credentials are stored in `credentials.json`.
- Uses **constant-time comparison** to prevent timing attacks.

### 2. **Rate Limiting**
- **Max Requests:** 100 requests per hour.
- Returns **429 Too Many Requests** if exceeded.

### 3. **Logging**
- Logs authentication attempts and request activities to `api_access.log`.

---

## Deployment
Run the API using Uvicorn:
```sh
uvicorn api:app --host 0.0.0.0 --port 8000
```

For production, use HTTPS and a process manager like **Gunicorn**.

---

## Error Handling
The API has a **global exception handler** to catch unexpected errors and return a JSON response with a `500` status.

