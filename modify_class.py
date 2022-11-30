import os

path = r"C:\Users\WIN\Desktop\碩士\deep learn\\train"  # 資料夾目錄
files = os.listdir(path)  # 得到資料夾下的所有檔名稱
s = []

for file in files:  # 資料夾所有檔案
    f = open(path + "/" + file, "r+", encoding="utf-8");  # 開啟所有檔案
    iter_f = iter(f);
    content = ""
    num_flag = 0
    for line in iter_f:
        lst = line.split(" ")
        lst[0] = "1"  # 改變第一個class數字
        for word in lst:  # 寫入的內容
            if num_flag == 0:
                content += word
                num_flag += 1
            else:
                content += " " + word
        num_flag = 0

    f.seek(0)  # 重置鼠標位置
    f.write(content)  # 寫入
