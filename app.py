import os, json, cv2, numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
import pytesseract

# pick up Tesseract path from env if set (Windows dev), else rely on PATH (/usr/bin/tesseract in Docker)
t_cmd = os.getenv("TESSERACT_CMD")
if t_cmd:
    pytesseract.pytesseract.tesseract_cmd = t_cmd

from IDCroper import CardExtractor
from DBHelper import SQLDatabase

app = FastAPI(title="Egypt ID OCR Service", version="0.1.0")

def _preprocess(img_bgr: np.ndarray, threshold: int) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binv = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY_INV)
    return cv2.bitwise_not(binv)

def _extract(img_bgr: np.ndarray, side: str, threshold: int):
    processed = _preprocess(img_bgr, threshold)
    extractor = CardExtractor(processed, img_bgr)  # adjust if your ctor differs
    res = extractor.getFront_IDData() if side == "F" else extractor.getBack_IDData()
    if isinstance(res, (dict, list)):
        return res
    try:
        return json.loads(res)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to parse extractor output")

@app.post("/recognize-text")
async def recognize_text(
    side: str = Query(..., pattern="^[FB]$", description="F=front, B=back"),
    threshold: int = Query(120, ge=0, le=255),
    image: UploadFile = File(...)
):
    data = await image.read()
    arr = np.frombuffer(data, dtype=np.uint8)  # correct for bytes
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image bytes")
    return JSONResponse(content=_extract(img, side, threshold))

@app.post("/save")
async def save_to_database(record: dict):
    server   = os.getenv('DATABASE_SERVER_IP')
    database = os.getenv('DATABASE_NAME')
    username = os.getenv('DATABASE_USERNAME')
    password = os.getenv('DATABASE_PASSWORD')

    try:
        db = SQLDatabase(server, database, username, password)
        db.connect()

        table = "IDs"
        if hasattr(db, "table_exists") and not db.table_exists(table):
            columns = {
                "Id": "INT IDENTITY(1,1) PRIMARY KEY",
                "Name": "NVARCHAR(200)",
                "Address": "NVARCHAR(400)",
                "DOB": "DATE",
                "NationalID": "VARCHAR(14)",
                "Gender": "NVARCHAR(10)",
                "EndDate": "DATE",
                "Profession": "NVARCHAR(200)",
                "MaritalStatus": "NVARCHAR(50)",
                "Religion": "NVARCHAR(50)"
            }
            db.create_table(table, columns)

        db.insert_record(table, record)
        return {"message": "Data saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
