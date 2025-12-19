# Face Recognition–Based Attendance Monitoring System

An AI-powered attendance tracking system that uses **computer vision and face recognition** to automatically identify students via webcam input and record attendance in real time. The system provides an interactive GUI for managing students, viewing attendance analytics, and sending automated email notifications.

---

## Overview

Manual attendance marking is time-consuming and error-prone. This project automates the process using **face recognition**, enabling fast, accurate, and paperless attendance tracking. Designed primarily for classroom use, the system supports student registration, real-time attendance capture, analytics, and manual overrides when required.

---

## Tech Stack

- **Language:** Python  
- **Computer Vision:** OpenCV (Haar Cascade, LBPH Face Recognizer)  
- **Web Interface:** Streamlit  
- **Data Handling:** Pandas, NumPy, CSV  
- **Visualization:** Plotly, st-aggrid  
- **Image Processing:** PIL  
- **Notifications:** SMTP (Email)  
- **Hardware:** Webcam  

---

## Key Features

- **Face Registration & Training**
  - Captures facial images via webcam
  - Trains an LBPH face recognition model
  - Stores student metadata securely in CSV format

- **Automated Attendance Tracking**
  - Detects faces in real time using Haar Cascade
  - Recognizes registered students and marks attendance
  - Prevents duplicate attendance entries for the same date

- **Interactive Dashboard**
  - View attendance by date, student, or overall summary
  - Display attendance data using tables and charts
  - Filter and explore records interactively

- **Manual Attendance Entry**
  - Allows instructors to add attendance manually when needed
  - Automatically updates records and analytics

- **Email Notifications**
  - Sends attendance confirmation emails upon successful check-in
  - Improves transparency and communication with stakeholders

---

## System Architecture / Workflow

1. **Student Registration**
   - Capture facial images using webcam
   - Store images and metadata
   - Train LBPH face recognizer

2. **Attendance Capture**
   - Detect faces using Haar Cascade
   - Recognize faces using trained LBPH model
   - Record attendance with timestamp

3. **Data Management & Analytics**
   - Store attendance records in CSV files
   - Process data using Pandas
   - Visualize insights using Plotly and interactive tables

---

## Results / Output

- Successfully identifies registered students in real time
- Records accurate attendance with timestamped entries
- Displays attendance statistics via charts and tables
- Sends automated email notifications upon attendance updates

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
2. Install required dependencies:
   ```bash
    pip install -r requirements.txt
3. Run the application:
  ```bash
    pip install -r requirements.txt
  ```
4. Ensure:
  - A working webcam is connected
  - Haar cascade XML file is present in the project directory

## Future Improvements

  - Replace CSV-based storage with a relational database
  - Improve recognition robustness under low-light conditions
  - Add role-based authentication for instructors
  - Integrate cloud deployment and centralized data storage

## Disclaimer

This project was developed for academic purposes. Face recognition accuracy may vary based on lighting conditions, camera quality, and dataset size.
