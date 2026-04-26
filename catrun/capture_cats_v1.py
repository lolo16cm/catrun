import cv2
import os

cat_name = input("Enter cat name: ")
os.makedirs(f'dataset/{cat_name}', exist_ok=True)

def open_imx477():
    # Use one single string, not tuple with ()
    pipeline = "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! nvvidconv flip-method=2 ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
    
    print(f"Pipeline: {pipeline}")
    
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            print("✅ IMX477 CSI camera opened!")
            return cap
    print("❌ CSI pipeline failed")
    return None

cap = open_imx477()
if cap is None:
    exit()

count = 0
print("Press SPACE to capture, Q to quit")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    cv2.putText(frame, f'Cat: {cat_name} | Saved: {count}',
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 255, 0), 2)

    cv2.imshow('Capture', frame)

    key = cv2.waitKey(1)
    if key == ord(' '):
        path = f'dataset/{cat_name}/{count}.jpg'
        cv2.imwrite(path, frame)
        count += 1
        print(f'✅ Saved: {path}')
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f'Done! {count} images saved for {cat_name}')
