import cv2
import json
from OCRExtractor import OCREngine
import re
import numpy as np
import base64

class CardExtractor:
    def __init__(self, Graycard_image,RGBCardImage):
        self.card_image = Graycard_image
        self.RGBCard=RGBCardImage
        self.card_width_mm = 85.6
        self.card_height_mm = 54.0
        self.pixels_per_mm_width = self.card_image.shape[1] / self.card_width_mm
        self.pixels_per_mm_height = self.card_image.shape[0] / self.card_height_mm

    def reduce_black_space(self, binary_image_path):
      # Load the binary image
      image = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)

      # Apply morphological operations to reduce black space
      kernel = np.ones((5,5), np.uint8)
      dilated = cv2.dilate(image, kernel, iterations=1)
      eroded = cv2.erode(dilated, kernel, iterations=1)

      # Perform connected component analysis
      num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded, connectivity=8)

      # Extract individual words based on bounding boxes
      words = []
      for i in range(1, num_labels):  # Skip background (label 0)
        x, y, w, h, _ = stats[i]
        word = image[y:y+h, x:x+w]

        # Resize word to have consistent height
        target_height = 32  # Adjust as needed
        target_width = int(w * target_height / h)  # Calculate target width based on aspect ratio
        word_resized = cv2.resize(word, (target_width, target_height))   
        words.append(word_resized)

      # Concatenate words horizontally to form a single sentence image
      sentence_image = np.concatenate(words, axis=1)

      return sentence_image

    def extract_data_area(self, dataTop, dataBottom, dataLeft, dataRight, mode="Gray"):
        top = int(dataTop * self.pixels_per_mm_height)
        bottom = self.card_image.shape[0] - int(dataBottom * self.pixels_per_mm_height)
        left = int(dataLeft * self.pixels_per_mm_width)
        right = self.card_image.shape[1] - int(dataRight * self.pixels_per_mm_width)
        if mode == "RGB":
            return self.RGBCard[top:bottom, left:right]
        
        return self.card_image[top:bottom, left:right]

    def save_data_area(self, data_area, output_path):
        cv2.imwrite(output_path, data_area)
        print("Data area extraction completed successfully.")

    def extractName(self, OCR):
        name_data = self.extract_data_area(13.0, 34.0, 30.0, 2.0)
        self.save_data_area(name_data, '1st_name_data_area.jpg')
        name = OCR.extract_arabic_text(name_data)
        name_data = self.extract_data_area(19.0, 29.0, 30.0, 2.0)
        self.save_data_area(name_data, '2nd_name_data_area.jpg')
        name2 = OCR.extract_arabic_text(name_data)
        return name+" "+name2
    
    def extractAddress(self, OCR):
        address_data = self.extract_data_area(25.0, 23.0, 30.0, 2.0)
        self.save_data_area(address_data, '1staddress_data_area.jpg')
        address = OCR.extract_arabic_text(address_data)

        address_data = self.extract_data_area(30.5, 18.0, 30.0, 2.0)
        self.save_data_area(address_data, '2ndaddress_data_area.jpg')
        address2 = OCR.extract_arabic_text(address_data)

        return address+" "+address2
    def extractID(self, OCR):
        id_data = self.extract_data_area(40.0, 5.0, 30.0, 2.0)
        self.save_data_area(id_data, 'id_data_area.jpg')
        ID = OCR.extract_numbers(id_data)
        return ID

    def getFront_IDData(self):
        OCR = OCREngine()
        All_data = self.extract_data_area(13.0, 18.0, 30.0, 2.0)
        self.save_data_area(All_data, 'Allfront_data_area.jpg')
        all=OCR.extract_arabic_text(All_data)
        file_path = "rtl_text.txt"
      # Open the file in write mode ('w') and specify encoding='utf-8' for handling Arabic text
        with open(file_path, 'w', encoding='utf-8') as file:
             file.write(all)
        # Split the text into lines and remove any empty lines
        lines = [line for line in all.split('\n') if line.strip()]
        print(lines)
        if len(lines) >= 4:
            # Extract the name and address
            name = lines[0] + ' ' + lines[1]
            address = lines[2] + ', ' + lines[3]
        else:
            name =self.extractName(OCR)
            address = self.extractAddress(OCR)
        ID=self.extractID(OCR)      
        DOB = self.extract_date_from_id(ID)

        face_data = self.extract_data_area(5.0, 20.0, 2.0, 60.0,"RGB")
        self.save_data_area(face_data, 'face_data_area.jpg')
        # Convert the image to base64
        _, buffer = cv2.imencode('.jpg', self.RGBCard)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        _, buffer = cv2.imencode('.jpg', face_data)
        encoded_face = base64.b64encode(buffer).decode('utf-8')

        # Store data in a dictionary
        data = {
            "name": name,
            "address": address,
            "ID": ID,
            "DOB": DOB,
            "image": encoded_image,
            "face": encoded_face
        }

        # Convert dictionary to JSON string
        json_data = json.dumps(data)
        return json_data
    
  
    def extract_date_from_id(self,id_number):
      # Extract millennium indicator, year, month, and day
      id = str(id_number).replace(' ', '')
      print("ID="+id)
      millennium_indicator = int(id[0])
      year = int(id[1:3])
      month = int(id[3:5])
      day = int(id[5:7])
      print("MIllinum="+str(millennium_indicator)+"year="+str(year),"month="+str(month), "dat="+str(day))
    
      # Determine the full year based on the millennium indicator
      if millennium_indicator == 2:
        year += 1900
      elif millennium_indicator == 3:
        year += 2000    
      # Return the date as a datetime object
      return  f"{year}-{month:02d}-{day:02d}" 
    
    def find_religion(self, text,gender):
       try:
        # Define regular expressions for the words "مسلم" and "مسيحي"
        muslim_pattern = re.compile(r'مسلم', re.IGNORECASE)
        christian_pattern = re.compile(r'مسيحى', re.IGNORECASE)
        # Split the text into lines and iterate through each line
        for line in text.split('\n'):
            # Search for the words in the line
            muslim_match = muslim_pattern.search(line)
            christian_match = christian_pattern.search(line)
            # Return the appropriate word if found in the text
            if muslim_match:
                if gender == 'm':
                    return "مسلم"
                elif gender == 'f':
                    return "مسلمة"
            elif christian_match:
                if gender == 'm':
                    return "مسيحي"
                elif gender == 'f':
                    return "مسيحية"
        
        # If no match found in any line, return None
        return None
       except Exception as e:
        print("Error:", e)
        return None
    def concatenate_images(img1, img2, img3):
        # Find contours in each binary image
        contours1, _ = cv2.findContours(img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours3, _ = cv2.findContours(img3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get bounding boxes of text regions in each image
        x1, y1, w1, h1 = cv2.boundingRect(contours1[0])
        x2, y2, w2, h2 = cv2.boundingRect(contours2[0])
        x3, y3, w3, h3 = cv2.boundingRect(contours3[0])

        # Determine dimensions of concatenated image
        max_width = max(x1 + w1, x2 + w2, x3 + w3)
        total_height = h1 + h2 + h3

        # Create blank concatenated image
        concatenated_img = np.zeros((total_height, max_width), dtype=np.uint8)

        # Paste each binary image into the concatenated image
        concatenated_img[y1:y1+h1, 0:w1] = img1
        concatenated_img[y2+h1:y2+h1+h2, 0:w2] = img2
        concatenated_img[y3+h1+h2:y3+h1+h2+h3, 0:w3] = img3

        return concatenated_img
    def find_gender(self, text):
     try:
        # Define regular expression patterns for male and female genders
        male_patterns = [
            re.compile(r'ذكر'),
            re.compile(r'ذ كر'),
            re.compile(r'دكري'),
            re.compile(r'دكر')
        ]

        female_patterns = [
            re.compile(r'أنثى'),
            re.compile(r'أنتى'),
            re.compile(r'انتى'),
            re.compile(r'آنثى'),
            re.compile(r'انثى')
        ]

        # Iterate through each line in the text
        for line in text.split('\n'):
            # Search for male patterns
            for pattern in male_patterns:
                male_match = pattern.search(line)
                if male_match:
                    print("Matched text:", male_match.group())
                    return "ذكر", 'm'

            # Search for female patterns
            for pattern in female_patterns:
                female_match = pattern.search(line)
                if female_match:
                    print("Matched text:", female_match.group())
                    return "أنثى", 'f'

        # If no match found, return empty strings
        return "", ""
     except Exception as e:     
        print("Error:", e)
        return "", ""
     
    def extractEndDate(self,OCR):
      enddate_data = self.extract_data_area(25.0, 23.0, 20.0, 40.0)
      yeardata=self.extract_data_area(25.0, 23.0, 20.0, 50.3)
      monthdata=self.extract_data_area(25.0, 23.0, 36.0, 45.5)
      daydata=self.extract_data_area(25.0, 23.0, 41.0, 40.5)
      self.save_data_area(monthdata, 'monthdata.jpg')
      month = OCR.extract_numbers(monthdata)
      monthstr=str(month).strip()
      self.save_data_area(daydata, 'daydata.jpg')
      day = OCR.extract_numbers(daydata)
      daystr=str(day).strip()
      self.save_data_area(yeardata, 'yeardata.jpg')
      year = OCR.extract_numbers(yeardata)
      yearstr=str(year).strip()
      print("yearstr"+yearstr)
      print("monthstr"+monthstr)
      print("daystr"+daystr)
      if yearstr.__len__() > 4:
        year = yearstr[0:4]
      
      if monthstr.__len__() > 2:
        month = monthstr[1:3]
      if daystr.__len__() > 2:
        day = daystr[1:3]
      self.save_data_area(enddate_data, 'enddate_data_area.jpg')
      enddate = str(year)+"-"+str(month)+"-"+str(day)
      return enddate 
    def get_last_two_digits(self,number):
        # Convert the number to a string
        number_str = str(number)
        # Get the last two characters
        if number_str.__len__() > 2:
            last_two_digits = number_str[1:]
        else:
            last_two_digits = number_str
        return last_two_digits
    def find_Mstatus(self, text, gender):
     try:
        if gender == 'm':
            single_patterns = [
                re.compile(r'أعزب'),
                re.compile(r'اعرب'),
                re.compile(r'اغزب'),
                re.compile(r'اغزب'),
                re.compile(r'اعزب'),
                re.compile(r'اعرب'),
                re.compile(r'أعرب'),               
                re.compile(r'أعزب'),  # Include optional ending for gender-neutral matching
                re.compile(r'عازب'),
                re.compile(r'عازب')  # Include optional ending for gender-neutral matching
            ]
            married_patterns = [
                re.compile(r'متزوج'),
                re.compile(r'متزوح'),
                re.compile(r'متروج'),  # Include optional ending for female gender
                re.compile(r'منروج'),
                re.compile(r'منروج')  # Include optional ending for female gender
            ]
            divorced_patterns = [
                re.compile(r'مطلق'),
                re.compile(r'مطلف'),
            ]
            widower_patterns = [
                re.compile(r'أرمل'),
                re.compile(r'ارمل'),
            ]
        elif gender=='f':
            single_patterns = [
                re.compile(r'عزباء'),
                re.compile(r'عزب'),  # Include optional ending for male gender
                re.compile(r'عزبة'),
                re.compile(r'عازبة|عازب')  # Include optional ending for gender-neutral matching
            ]
            married_patterns = [
                re.compile(r'متزوجة'),
                re.compile(r'متزوج|متزوجة'),  # Include optional ending for male gender
                re.compile(r'مزوجة'),
                re.compile(r'مزوج|مزوجة')  # Include optional ending for male gender
            ]
            divorced_patterns = [
                re.compile(r'مطلقة'),
                re.compile(r'مطلق')
            ]
            widower_patterns = [
                re.compile(r'أرملة'),
                re.compile(r'أرمل')
            ]

        # Iterate through each line in the text
        for line in text.split('\n'):
            # Search for single patterns
            for pattern in single_patterns:
                single_match = pattern.search(line)
                if single_match:
                    if gender == 'm':
                        print("Matched text:", single_match.group(),"gender="+gender)
                        return "أعزب"  
                    elif gender == 'f':
                        return "عزباء"

            # Search for married patterns
            for pattern in married_patterns:
                married_match = pattern.search(line)
                if married_match:
                    if gender == 'm':
                       return "متزوج" 
                    elif gender == 'f':
                       return "متزوجة"

            # Search for divorced patterns
            for pattern in divorced_patterns:
                divorced_match = pattern.search(line)
                if divorced_match:
                  if gender == 'm':
                       return "مطلق"
                  elif gender == 'f':
                       return "مطلقة"

            # Search for widower patterns
            for pattern in widower_patterns:
                widower_match = pattern.search(line)
                if widower_match:
                    if gender == 'm':
                       return "أرمل"
                    elif gender == 'f':
                       return "أرملة"

        # If no match found, return None
        return None
     except Exception as e:
        print("Error:", e)
        return None
    def getBack_IDData(self):
      OCR = OCREngine()
      All_data = self.extract_data_area(7.7, 22.0, 20.0, 17.0)
      self.save_data_area(All_data, 'All_data_area.jpg')
      All_ocr = OCR.extract_arabic_text(All_data)
      # Specify the file path where you want to save the text file
      file_path = "rtl_text.txt"
      # Open the file in write mode ('w') and specify encoding='utf-8' for handling Arabic text
      with open(file_path, 'w', encoding='utf-8') as file:
        file.write(All_ocr)

      # Extract profession data
      profession_data1 = self.extract_data_area(7.7, 42.0, 20.0, 16.0)
      profession_data2= self.extract_data_area(12.0, 37.0, 20.0, 17.0)

      self.save_data_area(profession_data1, 'profession1_data_area.jpg')
      self.save_data_area(profession_data2, 'profession2_data_area.jpg')
      profession1 = OCR.extract_arabic_text(profession_data1)
      profession2 = OCR.extract_arabic_text(profession_data2)
      profession=profession1+" "+profession2   
    
      # Extract religion data
      religion_data = self.extract_data_area(16.0, 33.0, 45.0, 30.0)
      gender_data =self.extract_data_area(16.0, 33.0, 60.0, 16.5)
      Mstatus_Data=self.extract_data_area(16.0, 33.0, 20.0, 45.0)
      self.save_data_area(religion_data, 'religion_data.jpg')
      self.save_data_area(Mstatus_Data, 'Mstatus_Data.jpg')
      self.save_data_area(gender_data, 'gender_data.jpg')
      CompinedDataArea = self.extract_data_area(16.0, 32.0, 20.0, 30.0)   
      CompinedDataArea2 = self.extract_data_area(16.0, 32.0, 45.0, 16.5)    
 
      self.save_data_area(CompinedDataArea, 'CompinedDataArea.jpg')
      self.save_data_area(CompinedDataArea2, 'CompinedDataArea2.jpg')

      combined=self.reduce_black_space('CompinedDataArea.jpg')
      combined2=self.reduce_black_space('CompinedDataArea2.jpg')

      self.save_data_area(combined, 'combinedCropedarea.jpg')
      self.save_data_area(combined2, 'combinedCropedarea2.jpg')

      compinedtext = OCR.extract_arabic_text(combined)
      compinedtext2 = OCR.extract_arabic_text(combined2)
      print('compinedtext2', compinedtext2)

      print('compinedtext', compinedtext)
      religion=OCR.extract_arabic_text(religion_data)   
      print('religion',religion)
      gender=OCR.extract_arabic_text(gender_data)
      print('gender',gender)
      marital_status=OCR.extract_arabic_text(Mstatus_Data)
      print('Mstatus',marital_status)
      genderChar='m'
      Gender,genderChar=self.find_gender(All_ocr)
      if Gender is None:
         Gender,genderChar=self.find_gender(compinedtext)
      if Gender is None:
         Gender,genderChar=self.find_gender(compinedtext2)   
      if Gender is None:
         Gender,genderChar=self.find_gender(gender)
      Religion=self.find_religion(All_ocr,genderChar)
      if Religion is None:
         Religion=self.find_religion(compinedtext,genderChar)
      if Religion is None:
         Religion=self.find_religion(compinedtext2,genderChar)   
      if Religion is None:
         Religion=self.find_religion(religion,genderChar)       
      Mstatus=self.find_Mstatus(All_ocr,genderChar) 
      if Mstatus is None:
         Mstatus=self.find_Mstatus(compinedtext,genderChar)   
      if Mstatus is None:
         Mstatus=self.find_Mstatus(compinedtext2,genderChar)   
      if Mstatus is None:
         Mstatus=self.find_Mstatus(marital_status,genderChar)  

      enddate=self.extractEndDate(OCR)

      if genderChar == 'f':
        # Extract husband's name data
        husband_name_data = self.extract_data_area(20.1, 29.0, 30.0, 16.5)
        self.save_data_area(husband_name_data, 'husband_name_data_area.jpg')
        husband_name = OCR.extract_arabic_text(husband_name_data)
      else :
        husband_name = ' '  

      
      # Convert the image to base64
      _, buffer = cv2.imencode('.jpg', self.RGBCard)
      encoded_image = base64.b64encode(buffer).decode('utf-8')
      print(husband_name)
      # Store data in a dictionary
      data = {
        'profession': profession,
        'religion': Religion,
        'gender': Gender,
        'marital_status': Mstatus,
        'enddate': enddate,
        'husband_name': husband_name,
        'image': encoded_image,
      }

      # Convert dictionary to JSON string
      json_data = json.dumps(data)
      return json_data


    
       
