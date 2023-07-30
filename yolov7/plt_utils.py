import  os

def openreadtxt(file_name):
    data = []
    file = open(file_name,'r')  #打开文件
    file_data = file.readlines() #读取所有行

    for row in file_data:
        tmp_list = row.split(' ') #按‘，'切分每行的数据
        tmp_list[-1] = tmp_list[-1].replace('\n', '')  # 去掉换行符
        tmp_list = [float(i) for i in tmp_list]
        data.append(tmp_list[1:]) #将每行数据插入data中
    return data

def get_file(path):
    ground_truth_dict = {}
    # 使用os.walk获取文件路径及文件
    for home, dirs, files in os.walk(path):
        # 遍历文件名
        for filename in files:
            # 将文件路径包含文件一同加入列表
            if filename.endswith('.txt'):
                cur = os.path.join(home,filename)
                v = openreadtxt(cur)
                ground_truth_dict[filename] = v
    return ground_truth_dict

if __name__ == '__main__':
    print(get_file(r'/home/ubuntu/lxd-workplace/xutao/yolov7/KneeDetJoint/labels/test/'))