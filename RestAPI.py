# RestAPI.py — FastAPI version

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
import cv2
import numpy as np
from IDCroper import CardExtractor
from DBHelper import SQLDatabase
import os, threading, time, json, queue

selectedtresh = 0
app = FastAPI(title="OCR Service", version="1.0.0")


# ---------------------------
# ------- Helpers -----------
def preprocess_image(card_image, char, scantype, tresh):
    tresh = int(tresh)
    gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
    if scantype == "Scanner":
        trsh = 145
    else:
        trsh = 120
    if char == 'B':
        trsh = trsh - 10
    _, thresh_img = cv2.threshold(
        gray, tresh, 255, cv2.THRESH_BINARY_INV + cv2.ADAPTIVE_THRESH_MEAN_C
    )
    final_image = cv2.bitwise_not(thresh_img)
    return final_image


def deskew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angle = 0.0
    if len(contours) > 0:
        rect = cv2.minAreaRect(contours[0])
        angle = rect[2]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def CropIDFromScannerImage(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.bitwise_not(gray)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    if max_contour is not None:
        (x, y, w, h) = cv2.boundingRect(max_contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cropped_image = image[y:y + h, x:x + w]
        cv2.imwrite('detected_card.jpg', image)
        cv2.imwrite('cropped_id_card.jpg', cropped_image)
        return cropped_image

    return None


def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))


def match_template(template, image_gray):
    best_match = None
    best_value = -1
    best_angle = 0
    best_template = template
    for angle in range(0, 360, 15):
        rotated_template = rotate_image(template, angle)
        result = cv2.matchTemplate(image_gray, rotated_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val > best_value:
            best_value = max_val
            best_match = max_loc
            best_template = rotated_template
            best_angle = angle
    return best_match, best_template.shape[::-1], best_angle


def draw_rectangle(image, top_left, width_height):
    bottom_right = (top_left[0] + width_height[1], top_left[1] + width_height[0])
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    return image, bottom_right


def crop_region(image, top_left, bottom_right):
    return image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]


def extract_id_card_From_ScannerImage(image):
    # adjust template path if needed
    template_path = r'Test/template.jpg'
    template_image = cv2.imread(template_path)
    if template_image is None:
        return None
    template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    top_left, template_size, _ = match_template(template_gray, gray)
    if top_left is None:
        return None

    scanned_image_with_rect, bottom_right = draw_rectangle(image, top_left, template_size)
    cv2.imwrite('scanned_image_with_rect.jpg', scanned_image_with_rect)
    cropped_id_card = crop_region(image, top_left, bottom_right)
    return cropped_id_card


def BeginProcessing(image, char, scantype, tresh):
    # returns JSON string on success; raises HTTPException on error
    # (keeps your original CardExtractor usage and JPEG artifacts)
    if scantype == "Scanner":
        card = CropIDFromScannerImage(image)
        if card is None:
            card = extract_id_card_From_ScannerImage(image)
    else:
        card = CropIDFromScannerImage(image)

    if card is None:
        raise HTTPException(status_code=400, detail="Could not detect ID card in the image.")

    if char == 'F':
        cv2.imwrite('Frontdetected_card.jpg', card)
        cv2.imwrite('FrontOriginal.jpg', image)
    elif char == 'B':
        cv2.imwrite('Backdetected_card.jpg', card)
        cv2.imwrite('BackOriginal.jpg', image)
    else:
        raise HTTPException(status_code=400, detail="Invalid char. Use 'F' for front or 'B' for back.")

    processed = preprocess_image(card, char, scantype, tresh)
    cv2.imwrite('processd_id.jpg', processed)

    extractor = CardExtractor(processed, card)
    if char == 'F':
        jsonstring = extractor.getFront_IDData()
    else:
        jsonstring = extractor.getBack_IDData()

    return jsonstring


def SaveTODataBase(record: dict):
    try:
        server = os.getenv('DATABASE_SERVER_IP', 'default_server_ip')
        database = os.getenv('DATABASE_NAME', 'default_database_name')
        username = os.getenv('DATABASE_USERNAME', 'default_username')
        password = os.getenv('DATABASE_PASSWORD', 'default_password')

        db = SQLDatabase(server=server, database=database, username=username, password=password)

        if not db.connection:
            db.connect()

        table_name = 'IDs'

        if not db.table_exists(table_name):
            # naive schema: NVARCHAR(MAX) for all keys in the record
            if isinstance(record, dict):
                columns = {key: "NVARCHAR(MAX)" for key in record.keys()}
                db.create_table(table_name, columns)

        db.insert_record(table_name, record)
        return True, "Data saved successfully"
    except Exception as e:
        return False, str(e)


# ---------------------------
# --------- Schemas ---------
class CheckFileRequest(BaseModel):
    directory_path: str
    side: str
    treshold: int


# ---------------------------
# --------- Routes ----------
@app.get("/", response_class=PlainTextResponse)
def home():
    return "OCR Server is running..."


@app.post("/recognize-text/{char}/{threshold}")
async def recognize_text(char: str, threshold: int, image: UploadFile = File(...)):
    try:
        data = await image.read()
        npimg = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        json_str = BeginProcessing(img, char, "Image", threshold)
        # extractor returns a JSON string — turn into dict for proper JSON response
        return JSONResponse(content=json.loads(json_str))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/save")
async def save_to_database(record: dict):
    ok, msg = SaveTODataBase(record)
    if ok:
        return {"message": msg}
    raise HTTPException(status_code=500, detail=msg)


@app.post("/check-file/")
async def check_file(req: CheckFileRequest):
    directory_path = req.directory_path
    side = req.side
    tresh = req.treshold

    if not os.path.isabs(directory_path):
        raise HTTPException(status_code=400, detail="Directory path must be absolute")

    if not os.path.exists(directory_path):
        # keep behavior: 200 with file_found False
        return {"file_found": False}

    result_queue = queue.Queue()

    def check_file_presence(dirpath, char, tresh, result_queue):
        timeout = 10
        start_time = time.time()
        while time.time() - start_time < timeout:
            if os.path.exists(dirpath):
                files = os.listdir(dirpath)
                for file in files:
                    file_path = os.path.join(dirpath, file)
                    if os.path.isfile(file_path):
                        image = cv2.imread(file_path, cv2.IMREAD_ANYCOLOR)
                        try:
                            json_str = BeginProcessing(image, char, "Scanner", tresh)
                            result_queue.put(("ok", json_str))
                        except Exception as e:
                            result_queue.put(("err", str(e)))
                        return
            time.sleep(1)
        result_queue.put(("err", "Timeout waiting for file"))

    thread = threading.Thread(target=check_file_presence, args=(directory_path, side, tresh, result_queue))
    thread.start()
    thread.join()

    if not result_queue.empty():
        status, payload = result_queue.get()
        if status == "ok":
            return JSONResponse(content=json.loads(payload))
        raise HTTPException(status_code=500, detail=payload)

    return {"thread_started": True, "result": "No result from thread"}
