#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install pytesseract
import cv2
import numpy as np
import pandas as pd
import pytesseract


# In[162]:


# Set Pytesseract configuration
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Load the image
image_path = 'C:/job hunting/nityo/table_image.png'
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny Edge Detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Use Hough Line Transform to find lines
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60, minLineLength=90, maxLineGap=3) #parameters need to be tuned to get correct lines

# Draw the lines on the image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imshow('Hough Lines', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[160]:


vertical_lines = []
horizontal_lines = []

# Loop over the lines and classify them into horizontal and vertical
for line in lines:
    x1, y1, x2, y2 = line[0]
    if abs(x2 - x1) > abs(y2 - y1): 
        horizontal_lines.append(line)
    else:  # vertical line
        vertical_lines.append(line)

# Function to calculate intersection point of two lines
def line_intersection(line1, line2):
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]
    
    px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / \
         ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4) + 1e-10)
    py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / \
         ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4) + 1e-10)
    return px, py

# Calculate intersection points
intersections = []
for h_line in horizontal_lines:
    for v_line in vertical_lines:
        px, py = line_intersection(h_line, v_line)
        intersections.append((px, py))
# Sort the intersection points
intersections = sorted(intersections, key=lambda p: (p[1], p[0]))

# Estimate the grid size
num_rows = len(horizontal_lines) - 1
num_cols = len(vertical_lines) - 1


# In[ ]:


# Function to chunk the sorted points into a grid
def chunk_it(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

# Split the intersections into rows and columns
grid = chunk_it(intersections, num_rows)


# In[ ]:


for row in grid:
    row.sort()  # Sort each row by the x coordinate to line up columns

# Get the size of the image
height, width = gray.shape

# Make sure the coordinate points are within the image boundaries
def within_bounds(coord, max_value):
    return max(0, min(int(coord), max_value))

table_data = []
# Loop through the grid, making sure not to go out of bounds
for i in range(len(grid) - 1):
    row_data = []
    for j in range(len(grid[i])):
        if j + 1 >= len(grid[i]) or i + 1 >= len(grid):
            # We are at the edge, no right or bottom line, so we cannot form a cell here
            continue
        top_left = grid[i][j]
        # Make sure the bottom_right point doesn't go out of the current row and next row bounds
        if j + 1 < len(grid[i + 1]):
            bottom_right = grid[i + 1][j + 1]
        else:
            # There is no bottom-right for this cell, it's not fully enclosed
            continue

        # Correct the coordinates to ensure they are within image bounds and are integers
        x1 = within_bounds(top_left[0], width)
        y1 = within_bounds(top_left[1], height)
        x2 = within_bounds(bottom_right[0], width)
        y2 = within_bounds(bottom_right[1], height)

        # Check if we have a valid cell rectangle
        if x1 < x2 and y1 < y2:
            cell = gray[y1:y2, x1:x2]
            text = pytesseract.image_to_string(cell, config='--psm 7').strip()
            row_data.append(text)
        else:
            row_data.append("Error in cell detection")  # Or append an empty string if preferred

    if row_data:
        table_data.append(row_data)


# In[ ]:


# Create DataFrame and save to Excel
df = pd.DataFrame(table_data)
excel_path = 'C:/job hunting/nityo/extracted_table.xlsx'
df.to_excel(excel_path, index=False)

