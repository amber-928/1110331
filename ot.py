import cv2

# 載入 cascade 模型
cascade = cv2.CascadeClassifier(r"C:\Users\user\Downloads\project_com\cascade.xml")

# 打開攝影機（或 IP 攝影機：使用網址）
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("無法連接到攝影機")
    exit()

# 取得影像尺寸
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 建立影片輸出物件
out = cv2.VideoWriter('detected_output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame_width, frame_height))

print("開始錄影，按 q 結束")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 灰階 + 對比強化
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # 偵測飲水機
    detections = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=120,
        minSize=(100, 200)
    )

    # 畫框
    for (x, y, w, h) in detections:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 顯示畫面
    cv2.imshow('Drink Machine Detection', frame)

    # ★ 寫入影片
    out.write(frame)

    # 按 q 結束
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
out.release()
cv2.destroyAllWindows()
print("錄影結束，影片已儲存為 detected_output.avi")
