# YoloV4 Darknet script

## 1. darknet_video_json.py

### Script介紹
此工具是為了計算IOU撰寫，參數如下所示請根據自己檔案路徑輸入，cmd則如下範例，如果沒有輸入參數則會依照程式default
### 使用方式

```
python darknet_video_json.py --input C:\Users\WIN\Desktop\碩士\test.mp4 
```
#### 1. --input 
影片路徑
#### 2. --weights 
Yolo Weight檔案路徑
#### 3. --config_file
Yolo cfg檔案路徑
#### 4. --data_file
Yolo data file路徑

## 2. image_txt.py
### Script介紹
將要訓練Yolo的圖片轉換成路徑變成一個txt，cmd則如下範例
### 路徑修改
- 圖片路徑 : 

https://github.com/HZM111368024/YoloV4_Script/blob/a31da36765b4e4d7e77dd66830182dc4ff91ac9b/image_txt.py#L3
https://github.com/HZM111368024/YoloV4_Script/blob/a31da36765b4e4d7e77dd66830182dc4ff91ac9b/image_txt.py#L10



### 使用方式
```
python image_txt.py 
```
## 3. modify_class.py
如果使用他人的dataset，可以使用這個script將要訓練的image的資訊txt中的class改變成自己想要的class number
### 路徑修改 
- 圖片資訊txt路徑 : 
https://github.com/HZM111368024/YoloV4_Script/blob/a31da36765b4e4d7e77dd66830182dc4ff91ac9b/image_txt.py#L3

### 使用方式
```
python modify_class.py
```