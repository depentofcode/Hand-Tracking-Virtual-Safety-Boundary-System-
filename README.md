# Hand-Tracking-Virtual-Safety-Boundary-System
A real-time computer vision system that detects hand movement and triggers alerts based on proximity to a virtual boundary using classical OpenCV techniques without machine learning models.
Purpose of the Project

The goal is to demonstrate a proof-of-concept (POC) system that:

Detects and tracks a hand in real-time

Displays a virtual on-screen boundary

Classifies interaction into:

 SAFE

 WARNING

 DANGER

Shows a strong alert message: "DANGER DANGER" when the hand is too close.

How It Works? 

The webcam frame is processed in real-time using OpenCV.

The hand is segmented using HSV skin-color detection.
The largest skin-colored contour is treated as the hand, while the face region is automatically removed to prevent false detection.

A fixed rectangle on the screen acts as a virtual safety boundary.

The distance between the hand’s centroid and the rectangle boundary determines the state:

State	Condition	UI Feedback
SAFE	Hand far (>150px)	Green box
WARNING	Near (60–150px)	Yellow box
DANGER	Very close / touching (<60px)	Red box + "DANGER DANGER"
 Technologies Used
Component	Details
Language	Python
Libraries	OpenCV, NumPy
Camera	Laptop/Webcam
Restrictions	No MediaPipe, No OpenPose, No Cloud APIs
 Installation
pip install opencv-python numpy

 Run Instructions (Jupyter Notebook)

Open the .ipynb file or copy the code into a Jupyter cell.

Ensure your webcam is not in use by another app.

Run the notebook cell.

Stop using:

Kernel → Interrupt

 Expected Output Behavior

When the hand moves, the box color and text update in real-time.

When too close, the display will show:

 DANGER DANGER 

 Performance

Runs on CPU only

Achieves ~8–20 FPS, depending on lighting and hardware.

 Limitations

Very strong/low lighting may reduce detection accuracy.

Skin-color-based tracking may fail with gloves or colored lighting.

 Conclusion

This project successfully demonstrates a working concept of:

Real-time hand tracking

Spatial awareness without advanced pose models

Dynamic safety alert visualization

It can be further extended for gesture control, AR interaction, or real industrial safety systems.
