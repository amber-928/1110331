import os
import cv2
from xml.dom import minidom
from os.path import basename

# 標記檔的路徑
xmlFolder = r"C:\Users\user\Downloads\project_com\x2"
# 原始圖片路徑
imgFolder = r"C:\Users\user\Downloads\project_com\positive"
projFolder = r"C:\Users\user\Downloads\project_com"

# 訓練圖片大小
outputSize = (40, 80)
# 產生的訓練圖片類型
imageKeepType = "jpg"
# 是否生成負樣本（去除標註區域的圖片）
generateNegativeSource = True

saveROIsPath = os.path.join(projFolder, "positives")
positiveDesc_file = os.path.join(projFolder, "positives.info")
negOutput = os.path.join(projFolder, "neg_bg")

totalLabels = 0
wLabels = 0
hLabels = 0

def saveROI(roiSavePath, imgFolder, xmlFilepath, labelGrep="", generateNeg=False):
    global totalLabels, wLabels, hLabels
    
    xml_filename, _ = os.path.splitext(xmlFilepath)
    xml_filename = basename(xml_filename)
    img_filename = xml_filename + "." + imageKeepType

    labelXML = minidom.parse(xmlFilepath)

    labelNames = []
    labelXstart = []
    labelYstart = []
    labelW = []
    labelH = []

    # 讀取標註內容
    for elem in labelXML.getElementsByTagName("name"):
        labelNames.append(elem.firstChild.data)
    for elem in labelXML.getElementsByTagName("xmin"):
        labelXstart.append(int(elem.firstChild.data))
    for elem in labelXML.getElementsByTagName("ymin"):
        labelYstart.append(int(elem.firstChild.data))
    for elem in labelXML.getElementsByTagName("xmax"):
        labelW.append(int(elem.firstChild.data))
    for elem in labelXML.getElementsByTagName("ymax"):
        labelH.append(int(elem.firstChild.data))

    image = cv2.imread(os.path.join(imgFolder, img_filename))
    if image is None:
        print(f"Cannot read image: {os.path.join(imgFolder, img_filename)}")
        return

    image2 = image.copy()

    countLabels = 0
    totalW = 0
    totalH = 0

    for i in range(len(labelNames)):
        if labelGrep == "" or labelGrep == labelNames[i]:
            countLabels += 1
            
            x1 = max(0, labelXstart[i])
            y1 = max(0, labelYstart[i])
            x2 = labelW[i]
            y2 = labelH[i]
            
            w = x2 - x1
            h = y2 - y1
            totalW += w
            totalH += h

            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                print(f"Warning: empty ROI for {xml_filename} label {i+1}")
                continue
            
            roi = cv2.resize(roi, outputSize)

            roiFile = os.path.join(roiSavePath, f"{xml_filename}_{countLabels}.{imageKeepType}")
            cv2.imwrite(roiFile, roi)

            # 產生負樣本：用黑色矩形蓋住標註區域
            if generateNegativeSource:
                cv2.rectangle(image2, (x1, y1), (x2, y2), (0, 0, 0), -1)

    if generateNegativeSource:
        negFile = os.path.join(negOutput, f"{xml_filename}_neg.{imageKeepType}")
        cv2.imwrite(negFile, image2)

    wLabels += totalW
    hLabels += totalH
    totalLabels += countLabels

    print(f"Found {countLabels} labels in {xml_filename}")

# 建立資料夾
os.makedirs(saveROIsPath, exist_ok=True)
if generateNegativeSource:
    os.makedirs(negOutput, exist_ok=True)

# 處理所有 XML
fileCount = 0
for file in os.listdir(xmlFolder):
    if file.endswith(".xml"):
        fileCount += 1
        print(f"Processing XML: {file}")
        xmlfile = os.path.join(xmlFolder, file)
        saveROI(saveROIsPath, imgFolder, xmlfile, labelGrep="", generateNeg=generateNegativeSource)

# 計算平均寬高
if totalLabels > 0:
    avgW = round(wLabels / totalLabels, 1)
    avgH = round(hLabels / totalLabels, 1)
else:
    avgW, avgH = 0, 0

with open(os.path.join(saveROIsPath, "desc.txt"), 'a') as f:
    f.write(f"{fileCount} XML files processed\n")
    f.write(f"Total labels: {totalLabels}\n")
    f.write(f"Average W:H = {avgW}:{avgH}\n")

print(f"----> Average W:H = {avgW}:{avgH}")

# 寫 positives.info 檔案
with open(positiveDesc_file, 'w') as f:
    for file in os.listdir(saveROIsPath):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            img = cv2.imread(os.path.join(saveROIsPath, file))
            if img is None:
                continue
            h, w = img.shape[:2]
            # OpenCV Haar training format: "path 1 0 0 w h"
            f.write(f"positives/{file} 1 0 0 {w} {h}\n")
import os

negOutput = r"C:\Users\liuta\Downloads\project\negative"
bg_txt_path = os.path.join(os.path.dirname(negOutput), "bg.txt")

with open(bg_txt_path, 'w') as f:
    for file in os.listdir(negOutput):
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            # 這裡寫相對路徑，或絕對路徑都可以，視訓練環境而定
            # 下面用相對於 bg.txt 所在路徑的相對路徑
            relative_path = os.path.join("neg_bg", file)
            f.write(relative_path + "\n")

print(f"bg.txt 已產生在 {bg_txt_path}")
