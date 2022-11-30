import os

path = r"C:\Users\WIN\Desktop\碩士\deep learn\\train"  # 資料夾目錄
files = os.listdir(path)  # 得到資料夾下的所有檔名稱
s = []
f = open("train.txt","w+",  encoding="utf-8");  # 開啟檔案
content = ""

for file in files:  # 資料夾所有檔案
    file_path = "C:\darknet-master\\build\darknet\\x64\data\\train\\" + file  # 資料夾路徑+檔案名稱
    content = content + file_path + "\n"  # 寫入內容

f.write(content)  # 寫入
