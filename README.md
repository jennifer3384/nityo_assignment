# nityo_assignment

# Assignment1
* Need to download `pytesseract` if not downloaded yet
download address: https://github.com/UB-Mannheim/tesseract/wiki

* Three parameters need to be tuned in function cv2.HoughLinesP():
1. threshold: The minimum number of intersections to detect a line. Lower this if you're not detecting enough lines.
2. minLineLength: The minimum length of a line. Increase this to avoid detecting short lines that aren't borders.
3. maxLineGap: The maximum allowed gap between points on the same line. Increase this to allow for breaks in the line (like gaps for text).
** Check the output image while tuning, to get borders fully detected.**
   
* Three path need to modify before running the code:
Pytesseract configuration(where tesseract.exe is stored), image storage, excel output

# Assignment2
* Make sure these three packages are installed: PyMuPDF, googletrans==4.0.0-rc1, tqdm
* Three path need to modify before running the code: Chinese pdf, English pdf, excel output 
