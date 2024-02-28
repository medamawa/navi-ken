import cv2
from ultralytics import YOLO

# カメラのキャプチャを開始
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('data/sample2.MOV')

# モデルのロード
model = YOLO("yolov8n-seg.pt")

# 面積検出フラグ
area_flg = False

while True:
    # カメラから画像を読み込み
    ret, frame = cap.read()

    if area_flg:
        cv2.putText(frame, 'WARNING!', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 15)
    else:
        cv2.putText(frame, 'SAFE', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 15)
    
    area_flg = False
    
    # 予測の実行
    results = model.predict(source=frame, show=True, line_width=1) 
 
    # 予測結果の取得
    for result in results:
        for id, xywhn in zip(result.boxes.cls, result.boxes.xywhn):
            area = xywhn[2] * xywhn[3]
            print(id, area)
            
            if (area > 0.5) and (id == 0):
                area_flg = True

    # キー入力を待つ
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# キャプチャを解放する
cap.release()

# ウィンドウをすべて閉じる
cv2.destroyAllWindows()
