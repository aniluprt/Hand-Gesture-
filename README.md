# Hand Gesture Recognition ✋🤚✌️👍

A real-time hand gesture recognition system using **OpenCV** and **MediaPipe**.  
It detects hand landmarks from webcam input and classifies gestures such as **Fist, Open Palm, Thumbs Up, and Victory** in real time.

---

## 🚀 Features
- Real-time webcam-based gesture recognition  
- Uses **MediaPipe Hands** for landmark detection  
- Classifies gestures into:
  - ✊ Fist  
  - ✋ Open Palm  
  - 👍 Thumbs Up  
  - ✌️ Victory  
  - ❓ Unknown (for unclassified poses)  
- Works with both **left and right hands**  

---

## 📂 Project Structure
```
hand-gesture-recognition/
│── gesture_recognition.py   # Main program
│── requirements.txt         # Dependencies
│── README.md                # Documentation
```

---

## 🛠️ Requirements
Install dependencies using:

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
opencv-python
mediapipe
```

---

## ▶️ How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/hand-gesture-recognition.git
   cd hand-gesture-recognition
   ```
2. Run the script:
   ```bash
   python gesture_recognition.py
   ```
3. Press **`q`** to quit the application.

---

## 📸 Demo
(Add a screenshot or GIF of your program running here, e.g., showing hand gestures being detected.)

---

## 📌 Future Improvements
- Add support for custom gesture training  
- Improve accuracy with more conditions  
- Export to mobile or web applications  

---

## 📝 License
This project is licensed under the MIT License.
