from flask import Flask, Response
import cv2

app = Flask(__name__)
cap = cv2.VideoCapture(0)

def get_camera():
    pipeline = (
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), "
        "width=1280, height=720, framerate=30/1 ! "
        "nvvidconv flip-method=2 ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink drop=1"
    )
    return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

def generate():
    while True:
        ret, frame = cap.read()
        _, buffer = cv2.imencode('.mp4', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' 
               + frame_bytes + b'\r\n')

@app.route('/video')
def video():
    return Response(generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''
    <html>
    <body style="background:black; text-align:center">
        <h2 style="color:white">Cat Robot Camera</h2>
        <img src="/video" width="640"/>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
