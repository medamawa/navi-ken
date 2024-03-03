import cv2
from ultralytics import YOLO
import numpy as np

# カメラのキャプチャを開始
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('data/sample2.MOV')

# モデルのロード
model = YOLO("yolov8n-seg.pt")

# 面積検出フラグ
global area_flg  # area_flg をグローバル変数として宣言
area_flg = False

def generate_frames():
    global area_flg

    red_color = [255, 0, 0] #[r,g,b]
    red_img_size = [400,400] #[height,width]
    red_img = create_monochromatic_img(red_color, red_img_size)
    red_img = cv2.cvtColor(red_img, cv2.COLOR_RGB2BGR)
    alpha = 0.65

    while True:
        # カメラから画像を読み込み
        ret, frame = cap.read()

        if area_flg:
            red_img = cv2.resize(red_img, (frame.shape[1], frame.shape[0]))
            frame = cv2.addWeighted(frame, alpha, red_img, 1 - alpha, 0)
            cv2.putText(frame, 'WARNING!', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 15)
        else:
            cv2.putText(frame, 'SAFE', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 15)
        
        area_flg = False
        
        # 予測の実行
        # results = model.predict(source=frame, show=True, line_width=1)
        results = model.predict(source=frame, show=False, line_width=1)

        # 予測結果の取得
        for result in results:
            for id, xywhn in zip(result.boxes.cls, result.boxes.xywhn):
                area = xywhn[2] * xywhn[3]
                print(id, area)
                
                if (area > 0.5) and (id == 0):
                    area_flg = True

        ret2, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
def create_monochromatic_img(color, size):
    r = color[0] * np.ones((size[1], size[0], 1), dtype=np.uint8)
    g = color[1] * np.ones((size[1], size[0], 1), dtype=np.uint8)
    b = color[2] * np.ones((size[1], size[0], 1), dtype=np.uint8)
    return np.concatenate([r, g, b], axis=2)