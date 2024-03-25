#!/usr/bin/env python
# coding: utf-8

# In[25]:


pip install pytesseract


# In[29]:


import cv2
import numpy as np
import pandas as pd
import pytesseract

# Pytesseract configuration
pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR'

# Load the image
image_path = 'C:/job hunting/nityo/table_image.png'
image = cv2.imread(image_path)

# Convert the image to gray scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Using Otsu's thresholding to binarize the image
_, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Denoising the image
denoised = cv2.fastNlMeansDenoising(binarized, None, 30, 7, 21)

# Use erosion and dilation to expose the main structure of the table
kernel = np.ones((2, 2), np.uint8)
eroded = cv2.erode(denoised, kernel, iterations=1)
dilated = cv2.dilate(eroded, kernel, iterations=1)

# Find contours
contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Extracting the bounding boxes for each contour
bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

# Here we will sort the bounding boxes and extract the data
# Since this is a general solution, we assume tables have a consistent structure across images
# The contours need to be sorted to read the data in a structured form

# Sort the bounding boxes [(x, y, w, h)] based on the y coordinate first and then x coordinate.
# This is necessary to read the table from top to bottom and left to right
bounding_boxes = sorted(bounding_boxes, key=lambda b: (b[1], b[0]))

# Since we are extracting a table, we need to identify rows and columns. 
# This can be a complex task and often requires custom logic based on the specific table structure.
# For this example, we're assuming the table is well structured with clear rows and columns.

# Initialize an empty list to hold each row's data
table_data = []

# Loop through the bounding boxes to extract text from each cell
for i, box in enumerate(bounding_boxes):
    # Extract the region of interest
    x, y, w, h = box
    roi = binarized[y:y+h, x:x+w]
    
    # Use pytesseract to convert image to string
    text = pytesseract.image_to_string(roi, config='--psm 6').strip()
    
    # Append the text to the row data list. For now, we just append everything in a single list.
    # You will need to implement logic to split these into different rows and columns.
    table_data.append(text)

# Assuming we have a single column table for simplicity as logic for rows/columns splitting 
# would require knowledge about the table layout which is unique to each table.

# Creating DataFrame and writing to an excel file
df = pd.DataFrame(table_data)
excel_path = 'C:/job hunting/nityo/extracted_table.xlsx'
df.to_excel(excel_path, index=False)


# In[31]:


import cv2
import numpy as np
import pandas as pd
from pytesseract import image_to_string

# Load the image
image_path = 'C:/job hunting/nityo/table_image.png'
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Binarize the image using Otsu's method
thresh, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Define a kernel for morphology operations
kernel = np.ones((2, 2), np.uint8)

# Use morphology to remove any small white noise
opening = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=2)

# Find the background area by dilating the image
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Finding sure foreground area (finding text areas)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Finding unknown region (this should be the lines of the grid)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1

# Now, mark the region of unknown with zero
markers[unknown == 255] = 0

# Apply watershed algorithm to find the boundaries of the objects
markers = cv2.watershed(image, markers)
image[markers == -1] = [255, 0, 0]

# Generate bounding boxes based on the identified markers
bounding_boxes = []
for label in np.unique(markers):
    if label == 0 or label == 1:
        continue
    # Create mask
    mask = np.zeros(gray.shape, dtype="uint8")
    mask[markers == label] = 255

    # Find contours and bounding boxes
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        bounding_boxes.append((x, y, w, h))

# Sort bounding boxes top to bottom, then left to right
bounding_boxes = sorted(bounding_boxes, key=lambda b: (b[1], b[0]))

# Extract data from the sorted bounding boxes
data = []
for (x, y, w, h) in bounding_boxes:
    # Crop the cell out of the binarized image
    cell = bin_img[y:y + h, x:x + w]

    # OCR the cell
    text = image_to_string(cell, config='--psm 7').strip()
    data.append(text)

# Since we are doing a general solution, we will attempt to estimate the number of columns in the table
# based on the x position of the bounding boxes. This is a rough heuristic and may not be perfect for all tables.
columns = sorted(np.unique([b[0] for b in bounding_boxes]))
num_columns = len(columns)
rows = [data[i:i + num_columns] for i in range(0, len(data), num_columns)]

# Save to an Excel file
df = pd.DataFrame(rows)
excel_path = 'C:/job hunting/nityo/extracted_table.xlsx'
df.to_excel(excel_path, index=False)


# In[ ]:




