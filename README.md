# Attendance_Manager

Here’s a well-formatted and easy-to-read version of your README steps:  

---

# Facial Recognition Attendance System  

## Setup Instructions  

### 📁 Initial Folder Setup  
Before running the program, create the following folders in your project directory:  
- **Attendance/** ➝ Stores attendance records.  
- **data/** ➝ Stores captured face data.  

### 🏗️ Collecting Dataset (`dataset.py`)  
1. Run `dataset.py`.  
2. Enter your name when prompted.  
3. The camera will capture **15 frames** of your face.  
4. Two `.pkl` (pickle) files will be generated and saved in the `data/` folder.  

### 📝 Taking Attendance (`attendance.py`)  
1. Run `attendance.py`.  
2. Press **'o'** to capture attendance.  
3. A new **CSV file** will be created in the `Attendance/` folder.  
