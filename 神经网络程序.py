import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Sampler, random_split
import matplotlib.pyplot as plt
import random

# 这是一个计时器
class Timer:
    def __init__(self):
        self.start_time = None
        self.pause_time = None
        self.total_paused = 0
        self.is_running = False
        self.is_paused = False
    
    def start(self):
        """开始计时"""
        if not self.is_running:
            self.start_time = time.time()
            self.is_running = True
            self.is_paused = False
            self.total_paused = 0
            print("计时器已开始")
        else:
            print("计时器已经在运行中")
    
    def pause(self):
        """暂停计时"""
        if self.is_running and not self.is_paused:
            self.pause_time = time.time()
            self.is_paused = True
            print("计时器已暂停")
        elif not self.is_running:
            print("计时器尚未开始")
        else:
            print("计时器已经暂停了")
    
    def resume(self):
        """继续计时"""
        if self.is_running and self.is_paused:
            # 计算暂停的时间并累加到总暂停时间中
            self.total_paused += time.time() - self.pause_time
            self.is_paused = False
            print("计时器已继续")
        elif not self.is_running:
            print("计时器尚未开始")
        else:
            print("计时器已经在运行中")
    
    def stop(self):
        """停止计时并返回总经过时间"""
        if self.is_running:
            end_time = time.time()
            elapsed = end_time - self.start_time - self.total_paused
            if self.is_paused:
                # 如果当前处于暂停状态，需要减去当前暂停的时间
                elapsed -= (time.time() - self.pause_time)
            
            self.is_running = False
            self.is_paused = False
            print(f"计时器已停止，总用时: {elapsed:.2f} 秒")
            return elapsed
        else:
            print("计时器尚未开始")
            return 0
    
    def get_elapsed_time(self):
        """获取当前经过的时间（不包括暂停时间）"""
        if self.is_running:
            if self.is_paused:
                # 如果暂停了，返回从开始到暂停的时间减去总暂停时间
                elapsed = self.pause_time - self.start_time - self.total_paused
            else:
                # 如果正在运行，返回当前时间减去开始时间和总暂停时间
                elapsed = time.time() - self.start_time - self.total_paused
            return elapsed
        else:
            return 0
    
    def reset(self):
        """重置计时器"""
        self.__init__()
        print("计时器已重置")

# 这是一个读取加速度信息用的简单类
class Acc:
    def __init__(self, cond=False, healthy_num=None, node_num=None):
        self.cond = cond
        self.healthy_num = healthy_num if healthy_num is not None else []
        self.node_num = node_num if node_num is not None else []

# 这是一个负责将文件名转为数字的类，即将健康状态转换为对应的数字编号
class LabelMapper:
    def __init__(self):
        # 定义所有可能的标签
        self.a_labels = [f"A{i}.xlsx" for i in range(10)]  # A0到A9
        self.b_labels = [f"B{i}.xlsx" for i in range(1, 12)]  # B1到B11
        
        # 合并所有标签并创建映射
        self.all_labels = self.a_labels + self.b_labels
        self.str_to_num = {label: idx for idx, label in enumerate(self.all_labels)}
        self.num_to_str = {idx: label for label, idx in self.str_to_num.items()}
    
    # 将字符串标签映射为数字标签
    def str_to_num_map(self, str_label):
        return self.str_to_num.get(str_label, -1)  # 如果找不到返回-1
    
    # 将数字标签映射回字符串标签
    def num_to_str_map(self, num_label):
        return self.num_to_str.get(num_label, "Unknown")  # 如果找不到返回"Unknown"


# 这是一个专门用于画损失函数图像的类
class plot_loss:
    def __init__(self):
        self.train_loss = []
        self.val_loss = []
    
    def Add_trainloss(self, x):
        self.train_loss.append(x)

    def Add_valloss(self, x):
        self.val_loss.append(x)

    def Plot(self, show_mode = True, grid = False, save_mode = False, png_name = "TrainLoss"):
        png_name = png_name + "_loss.png"
        epoch = list(range(1, len(self.train_loss) + 1))
        plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定默认字体
        plt.figure(figsize=(5, 3))
        plt.plot(epoch, self.train_loss, color='black', linestyle='-', linewidth=1, label='训练集')
        plt.plot(epoch, self.val_loss, color='red', linestyle='--', linewidth=1, label='验证集')
        plt.xlabel('训练轮数', fontsize=10)
        plt.ylabel('损失函数值', fontsize=10)
        # plt.title('Loss Curve')
        plt.legend(fontsize=10)
        plt.grid(grid)      # 是否显示网格
        plt.tight_layout()  # 确保所有元素（包括文本）完整显示
        if save_mode == True:
            plt.savefig(png_name)  # 保存图像
        plt.show()
        if show_mode == False:
            plt.close()  # 关闭图形

    # 绘制指定训练轮数范围的图像（用于局部放大图片）
    def Plot_big(self, epoch_begin, epoch_end, show_mode = True, grid = False, save_mode = False, png_name = "TrainLoss"):
        png_name = png_name + "_loss" + str(epoch_begin) + "-" + str(epoch_end) + ".png"
        epoch = list(range(epoch_begin, epoch_end))
        train_loss_temp = self.train_loss[epoch_begin + 1:epoch_end + 1]
        val_loss_temp = self.val_loss[epoch_begin + 1:epoch_end+1]
        plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定默认字体
        plt.figure(figsize=(5, 3))
        plt.plot(epoch, train_loss_temp, color='black', linestyle='-', linewidth=1, label='训练集', marker='o', markersize=5)
        plt.plot(epoch, val_loss_temp, color='red', linestyle='--', linewidth=1, label='验证集', marker = '*', markersize=5)
        plt.xlabel('训练轮数', fontsize=10)
        plt.ylabel('损失函数值', fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(grid)      # 是否显示网格
        plt.tight_layout()  # 确保所有元素（包括文本）完整显示
        if save_mode == True:
            plt.savefig(png_name)  # 保存图像
        plt.show()
        if show_mode == False:
            plt.close()  # 关闭图形

class plot_acc_rate:
    def __init__(self):
        self.train_acc_rate = []
        self.val_acc_rate = []
    
    def Add_train_acc_rate(self, x):
        self.train_acc_rate.append(x)

    def Add_val_acc_rate(self, x):
        self.val_acc_rate.append(x)
        

    def Plot(self, show_mode = True, grid = False, save_mode = False, png_name = "TrainLoss"):
        png_name = png_name + "_acc_rate.png"
        epoch = list(range(1, len(self.val_acc_rate) + 1))
        plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定默认字体
        plt.figure(figsize=(5, 3))
        plt.plot(epoch, self.train_acc_rate, color='black', linestyle='-', linewidth=1, label='训练集')
        plt.plot(epoch, self.val_acc_rate, color='red', linestyle='--', linewidth=1, label='验证集')
        plt.xlabel('训练轮数', fontsize=10)
        plt.ylabel('准确率', fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(grid)      # 是否显示网格
        plt.tight_layout()  # 确保所有元素（包括文本）完整显示
        if save_mode == True:
            plt.savefig(png_name)  # 保存图像
        plt.show()
        if show_mode == False:
            plt.close()  # 关闭图形

    # 绘制指定训练轮数范围的图像（用于局部放大图片）
    def Plot_big(self, epoch_begin, epoch_end, show_mode = True, grid = False, save_mode = False, png_name = "TrainLoss"):
        png_name = png_name + "_acc_rate" + str(epoch_begin) + "-" + str(epoch_end) + ".png"
        epoch = list(range(epoch_begin, epoch_end))
        acc_rate_temp = self.acc_rate[epoch_begin + 1:epoch_end + 1]
        plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定默认字体
        plt.figure(figsize=(5, 3))
        plt.plot(epoch, acc_rate_temp, color='red', linestyle='--', linewidth=1, label='验证集', marker='^', markersize=6)
        plt.xlabel('训练轮数', fontsize=10)
        plt.ylabel('准确率', fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(grid)      # 是否显示网格
        plt.tight_layout()  # 确保所有元素（包括文本）完整显示
        if save_mode == True:
            plt.savefig(png_name)  # 保存图像
        plt.show()
        if show_mode == False:
            plt.close()  # 关闭图形

# 这个函数负责加速度信息的预处理(读取+计算)，在主函数(main)中只需要调用该函数即可完成
def Acc_prep():

    A = Acc()
    B = Acc()
    str_temp = ""
    # 获取用户输入，并转换为布尔值
    #str_temp = input("是否需要读取单个螺栓松动的健康状态数据(A0-A9)(是：1，否：2): ")
    print("是否需要读取单个螺栓松动的健康状态数据(A0-A9)(是：1，否：2): \n1")
    str_temp = "1"
    A.cond = str_temp.lower() in ['是', '1', 'yes']
    if A.cond:
        #str_temp = input("请输入需要读取的健康状态编号(0-9)：") # 计算相关函数矩阵所需要的观测点编号
        print("请输入需要读取的健康状态编号(0-9)：\n0 1 2 3 4 5 6 7 8 9")
        str_temp = "0 1 2 3 4 5 6"
        #str_temp = "0 1"
        #str_temp = "0 1"
        A.healthy_num = list(map(int, str_temp.split()))
        #str_temp = input("请输入观测点编号(1-6)：") # 计算相关函数矩阵所需要的观测点编号
        print("请输入观测点编号(1-6)：\n1 2 3 4 5 6")
        str_temp = "1 2 3 4 5 6"
        A.node_num = list(map(int, str_temp.split()))
    
    #str_temp = input("是否需要读取组合螺栓松动的健康状态数据(B1-B11)(是：1，否：2): ")
    print("是否需要读取组合螺栓松动的健康状态数据(B1-B11)(是：1，否：2): \n2")
    str_temp = "2"
    B.cond = str_temp.lower() in ['是', '1', 'yes']
    if B.cond:
        str_temp = input("请输入需要读取的健康状态编号(1-11)：") # 计算相关函数矩阵所需要的观测点编号
        B.healthy_num = list(map(int, str_temp.split()))
        str_temp = input("请输入观测点编号(1-6)：") # 计算相关函数矩阵所需要的观测点编号
        B.node_num = list(map(int, str_temp.split()))
    
    file_name = FileName(A, B)

    return xlsx_to_ipm(file_name, A, B), A, B

# 得到所有加速度信息文件的名称，是Acc_prep中的一个子函数
def FileName(A, B):
    file_name = []
    name_temp = ""

    if A.cond:
        for i in A.healthy_num:
            name_temp = 'A' + str(i) + ".xlsx"
            file_name.append(name_temp)
    if B.cond:
        for i in B.healthy_num:
            name_temp = 'B' + str(i) + ".xlsx"
            file_name.append(name_temp)
        
    return file_name

# 这个函数负责加速度读取+计算，是Acc_prep中的一个子函数
def xlsx_to_ipm(file_name, A, B):
    path_D = 'D:\\试验数据_2025_09_11\\' # 数据所在文件的路径
    K = 600 # 划分数据子集的数量，标签数据库的容量
    vector_length = 500 # 每个向量的长度
    num_columns = vector_length * K # 需要读取的总行数

    label_map = LabelMapper()
    acc_ipm_all_list = []
    acc_labels_all_list = []

    for name in file_name:
        df = pd.read_excel(path_D + name, header=None, usecols='A:G',skiprows=150000, nrows=num_columns)
        
        data_columns = df.columns[1:]  # B-F列
        
        # 初始化向量列表
        vectors = []
        
        # 对每列数据分别进行处理
        for col in data_columns:
            # 获取当前列数据
            col_data = df[col].values
            
            # 将列数据重塑为K个向量
            col_vectors = col_data.reshape(K, vector_length)
            
            # 添加到总向量列表
            vectors.append(col_vectors)
        
        # 将所有向量转换为NumPy数组
        acc_vectors= np.array(vectors)
        
        # 计算内积矩阵
        acc_ipm_temp = torch.zeros(K, 6, 6)
        acc_labels_temp = torch.zeros(K, dtype=torch.long)
        for i in range(K):
            for m in range(6):
                for n in range(m + 1):
                    acc_ipm_temp[i][m][n] = acc_vectors[m][i] @ acc_vectors[n][i]
                    acc_ipm_temp[i][n][m] = acc_vectors[m][i] @ acc_vectors[n][i]
                    acc_labels_temp[i] = label_map.str_to_num_map(name)
        acc_ipm_all_list.append(acc_ipm_temp)
        acc_labels_all_list.append(acc_labels_temp)

    acc_ipm_all = torch.cat(acc_ipm_all_list, dim = 0)
    acc_labels_all = torch.cat(acc_labels_all_list, dim = 0)

    return acc_ipm_all, acc_labels_all

#--------------------神经网络--------------------

# 这个函数负责完成训练集，测试集，验证集的划分
def Data_split(acc_cfm, acc_labels, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    acc_dataset = TensorDataset(acc_cfm, acc_labels)

    train_size = int(train_ratio * len(acc_dataset))
    val_size   = int(val_ratio   * len(acc_dataset))
    test_size  = int(test_ratio  * len(acc_dataset))
    
    # 划分
    acc_train_dataset, acc_val_dataset, acc_test_dataset = random_split(
        acc_dataset, [train_size, val_size, test_size]
    )

    acc_train_dataloader = DataLoader(acc_train_dataset, batch_size=32, shuffle=True)
    acc_val_dataloader = DataLoader(acc_val_dataset, batch_size=32, shuffle=False)
    acc_test_dataloader = DataLoader(acc_test_dataset, batch_size=32, shuffle=False)

    return acc_train_dataloader, acc_val_dataloader, acc_test_dataloader


class ConvNet(nn.Module):
    def __init__(self, num_classes=7):
        super(ConvNet, self).__init__()
        
        # 第一层卷积块
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, 
                out_channels=8, 
                kernel_size=3,   # 卷积核大小为3*3
                stride=1,        # 步长为1
                padding=1),
            nn.ReLU(inplace=True)  # inplace=True可以节省内存
        )
        
        self.batch_norm = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(
            kernel_size=2,        # 核尺寸为2*2
            stride=2)             # 步长2
        self.flatten = nn.Flatten()  # !!!!!此处可以更改尝试从第0维度展开!!!!!
        self.fc = nn.Linear(72, num_classes)
        
        
    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x: 输入张量, 形状应为 (batch_size, 1, H, W)
        
        返回:
            output: 输出张量, 形状为 (batch_size, num_classes)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)  # 增加通道维度，从(32,6,6)变化为(32,1,6,6)
        # 第一层卷积块
        x = self.conv_block1(x)   # 输出: (batch_size, 16, H, W)
        
        # 批归一化
        x = self.batch_norm(x)    # 输出: (batch_size, 32, H, W)
        
        # 最大池化
        x = self.pool(x)          # 输出: (batch_size, 32, H/2, W/2)
        
        # 展平
        x = self.flatten(x)       # 输出: (batch_size, 32 * (H/2) * (W/2))
        
        # 全连接层
        x = self.fc(x)            # 输出: (batch_size, num_classes)
        
        return x

def Net_train(train_loader, val_loader, net_name):  
    model = ConvNet(num_classes=10)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 200
    train_plot = plot_loss()
    acc_plot = plot_acc_rate()
    for epoch in range(num_epochs):
        # ----- 训练阶段 -----
        model.train()  # 确保切换回训练模式
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            
            #print("训练数据原始形状:", batch_X.shape)
            #batch_X = batch_X.unsqueeze(1)

            # 前向传播
            pred = model(batch_X)
            loss = loss_fn(pred, batch_y)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
        # 计算平均训练损失
        train_loss = train_loss / len(train_loader.dataset)
        train_plot.Add_trainloss(train_loss)
        
        # ----- 验证阶段 -----
        # 补充计算一个在训练集上的准确率
        train_acc = Net_test(model, train_loader)
        acc_plot.Add_train_acc_rate(train_acc)

        model.eval()  # 切换为评估模式
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():  # 禁用梯度计算
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = loss_fn(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        # 计算验证指标
        val_loss = val_loss / len(val_loader.dataset)
        train_plot.Add_valloss(val_loss)
        val_acc = correct / total
        acc_plot.Add_val_acc_rate(val_acc)
            
        # 打印进度
        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f} | '
              f'Val Acc: {val_acc:.4f}')

    #----------画图----------    
    #------------------------
    train_plot.Plot(grid = True, save_mode = True, png_name = net_name)
    acc_plot.Plot(grid = True, save_mode = True, png_name = net_name)
    #------------------------
    #------------------------

    # 训练结束后的模型
    torch.save(model, net_name)
    return model

def Net_test(net_model, test_dataloader):
    net_model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    i = 0
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            outputs = net_model(inputs)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Loss: {test_loss/len(test_dataloader):.4f}, Test Acc: {100 * correct / total:.2f}%")
    return correct / total

def Net_loader(net_name):
    model = torch.load(net_name, weights_only=False)
    return model

# 程序主函数
def main():
    read_cond = True
    timer = Timer()
    cnn_i = 0
    while True:
        # 记录开始时间
        timer.start()

        seed_input = input("输入随机数种子: ")
        torch.manual_seed(seed_input)

        user_input = input("是否需要重新读取加速度数据: ")
        read_cond = user_input.lower() in ['是', '1', 'yes']

        # 读取加速度信息，并完成内积矩阵的计算
        if read_cond:
            (acc_ipm, acc_labels), A, B= Acc_prep()
            read_cond = False
            print("Acc_prep is OK!")
            # time.sleep(5)

        # 根据随机数种子生成对应网络名称
        net_name = str(cnn_i) + "_" + seed_input + "_cnn.pt"       
        cnn_i += 1

        # 划分训练集，验证集和测试集
        train_dataloader, val_dataloader, test_dataloader = Data_split(acc_ipm, acc_labels)
        print("OK!!!!!")

        # 训练神经网络，并保存
        Net_train(train_dataloader, val_dataloader, net_name)
        print("Net_train is OK!")

        # 读取神经网络
        net_model = Net_loader(net_name)
        print("Net_loader is OK!")

        # 测试神经网络
        Net_test(net_model, test_dataloader)
        print("Net_test is OK!")

        # 记录结束时间，并计算耗时
        timer.stop()

main()



