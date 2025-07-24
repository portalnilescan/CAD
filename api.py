import os
import json
import time
import secrets
import logging
from typing import Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
from pydantic import BaseModel, Field
from AI_model import classify  # Assuming classify is your OCR classification function
from fastapi.responses import FileResponse


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="api_access.log"
)
logger = logging.getLogger("IDCardAPI")

app = FastAPI(
    title="MedVerse AI Engine",
    description="API for Dicom Processing",
    version="1.0.0"
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your specific frontend domains for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to credentials file
CREDENTIALS_FILE = "credentials.json"

# Route to serve the HTML form
@app.get("/", include_in_schema=False)
async def serve_form():
    """
    Serve the HTML form for ID card recognition.
    """
    html_file_path = os.path.join(os.path.dirname(__file__), "form.html")
    if os.path.exists(html_file_path):
        return FileResponse(html_file_path)
    else:
        logger.error("form.html file not found")
        raise HTTPException(status_code=404, detail="form.html not found")
    
# Rate limiting settings
RATE_LIMIT_REQUESTS = 100  # Maximum requests per time window
RATE_LIMIT_WINDOW = 3600  # Time window in seconds (1 hour)
request_history = {}

# Load API credentials from JSON file
def load_credentials() -> Dict[str, str]:
    if not os.path.exists(CREDENTIALS_FILE):
        logger.error(f"Credentials file {CREDENTIALS_FILE} not found")
        raise HTTPException(status_code=500, detail="Server credentials not configured")

    with open(CREDENTIALS_FILE, "r") as f:
        try:
            credentials = json.load(f)
        except json.JSONDecodeError:
            logger.error("Invalid JSON in credentials file")
            raise HTTPException(status_code=500, detail="Invalid credentials configuration")

    if not all(key in credentials for key in ["api_key", "api_secret"]):
        logger.error("Invalid credentials format")
        raise HTTPException(status_code=500, detail="Invalid credentials configuration")

    return credentials

# Rate limiting check
def check_rate_limit(client_id: str = "global") -> bool:
    current_time = time.time()

    if client_id not in request_history:
        request_history[client_id] = []

    # Clean old requests
    request_history[client_id] = [
        timestamp for timestamp in request_history[client_id]
        if current_time - timestamp < RATE_LIMIT_WINDOW
    ]

    # Check if rate limit exceeded
    if len(request_history[client_id]) >= RATE_LIMIT_REQUESTS:
        logger.warning(f"Rate limit exceeded for client_id: {client_id}")
        return False

    # Add current request timestamp
    request_history[client_id].append(current_time)
    return True

# Authentication dependency
async def verify_api_key(
    x_api_key: str = Header(...),
    x_api_secret: str = Header(...)
):
    logger.info("Authentication attempt")

    if not check_rate_limit():
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again later."
        )

    credentials = load_credentials()
    if not (
        secrets.compare_digest(x_api_key, credentials["api_key"]) and
        secrets.compare_digest(x_api_secret, credentials["api_secret"])
    ):
        logger.warning("Invalid API credentials")
        raise HTTPException(status_code=403, detail="Invalid API credentials")

    logger.info("Authentication successful")
    return {"authenticated": True}

# Pydantic model for the response
class IDResponse(BaseModel):
    result: int
    patient_id: str
    StudyInstanceUID: str
    finding: list[dict] = Field(default_factory=list, description="List of findings from OCR classification")

@app.post("/api/id-recognition", response_model=IDResponse)
async def process_id_card(
    file: UploadFile = File(...),
    auth_info: Dict[str, Any] = Depends(verify_api_key)
):
    """
    Process ID card image and return recognition results.
    """
    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        logger.info(f"File saved temporarily at {temp_file_path}")

        try:
            # Process the file using the classification function
            scan_result = classify(temp_file_path)

            # Validate the response structure
            if not all(key in scan_result for key in ["result", "patient_id", "StudyInstanceUID", "finding"]):
                raise ValueError("Invalid response structure from the OCR classification")

            logger.info("Processing successful")
            return IDResponse(**scan_result)

        finally:
            # Ensure the temporary file is removed
            os.unlink(temp_file_path)
            logger.info(f"Temporary file {temp_file_path} deleted")

    except Exception as e:
        logger.error(f"Error processing the file: {e}")
        raise HTTPException(status_code=500, detail="Error processing ID card")

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred"}
    )

# Run the application
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting the server...")
    uvicorn.run(app, host="0.0.0.0", port=8001)

    from fastapi.responses import FileResponse

