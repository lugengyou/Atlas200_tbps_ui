'''
Author: gengyou.lu 1770591868@qq.com
Date: 2025-01-07 10:34:13
FilePath: /Atlas200_tbps_ui/ui_designer/TbpsUiMainWindow.py
LastEditTime: 2025-01-08 16:43:04
Description: tbps ui main window
'''
import os
import sys
# 通过当前文件目录的相对路径设置工程的根目录
current_file_path = os.path.abspath(os.path.dirname(__file__))
project_base_path = os.path.abspath(os.path.join(current_file_path, "../"))

sys.path.append(project_base_path)
from deploy.deploy_tbps import tokenize, transfer_pic, net
from deploy.simple_tokenizer import SimpleTokenizer


import json
import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow,QApplication,QWidget,QGraphicsScene
from .Ui_tbps import Ui_MainWindow 


class MyMainWindow(QMainWindow,Ui_MainWindow):
    def __init__(self,parent =None):
        super(MyMainWindow,self).__init__(parent)
        self.setupUi(self)
        # 初始化模型        
        if 0:
            self.image_encoder = net(os.path.join(project_base_path, "deploy/model/xsmall_image_encode_310B4.om"))
            bpe_path = os.path.join(project_base_path, "data/bpe_simple_vocab_16e6.txt.gz")
            self.tokenizer = SimpleTokenizer(bpe_path)
            self.text_encoder = net(os.path.join(project_base_path, "deploy/model/xsmall_text_encode_310B4.om")) 
            self.consine_sim_model = net(os.path.join(project_base_path, "deploy/model/similarity_310B4.om")) 
        else:
            self.image_encoder = None
            self.tokenizer = None
            self.text_encoder = None
            self.consine_sim_model = None

        # 静态检索相关变量
        self.static_database_file_path = ""
        self.static_database_json_file_path = ""
        
        # 动态检索相关变量
        self.dynamic_database_base_path = "" # 所选数据集文件夹的父目录，用于构建图像完整路径
        self.dynamic_dataset_folder_name = "" # 所选数据集文件夹名称，用于构建存储图像特征文件名
        self.dynamic_database_image_files = []
        self.dynamic_image_features = None
        
        # 显示相关变量
        self.show_images_label_list = [self.label_show_img1, self.label_show_img2, self.label_show_img3, self.label_show_img4, self.label_show_img5,
                                          self.label_show_img6, self.label_show_img7, self.label_show_img8, self.label_show_img9, self.label_show_img10]
        self.show_sim_label_list = [self.label_show_sim1, self.label_show_sim2, self.label_show_sim3, self.label_show_sim4, self.label_show_sim5,
                                    self.label_show_sim6, self.label_show_sim7, self.label_show_sim8, self.label_show_sim9, self.label_show_sim10]

    def closeEvent(self, event):
        self.release_resources()
        event.accept()  # 接受关闭事件

    def release_resources(self):
        print("Release resources")

    # ************************ slot functions ************************ #
    def slot_select_dataset(self):        
        # 设置基础路径
        current_file_path = os.path.abspath(os.path.dirname(__file__))
        project_base_path = os.path.abspath(os.path.join(current_file_path, "../data/static_database"))
        print(project_base_path)
        # 打开文件选择对话框
        static_database_file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '选择文件', project_base_path)
        if static_database_file_path:
            self.lineEdit_select_dataset.setText(static_database_file_path)  # 设置选择的文件路径到 QLineEdit

    def slot_select_path(self):
        # 设置基础路径
        current_file_path = os.path.abspath(os.path.dirname(__file__))
        project_base_path = os.path.abspath(os.path.join(current_file_path, "../data/dynamic_database"))
        # print(project_base_path)
        # 打开文件选择对话框
        dynamic_database_path = QtWidgets.QFileDialog.getExistingDirectory(self, '选择文件夹', project_base_path)
        if dynamic_database_path:
            self.lineEdit_select_path.setText(dynamic_database_path)

    def slot_search(self):
        self.terminal_message("=========== Start Search ===========")
        # 获取输入文本描述
        enter_text_description = self.textEdit_enter_text_description.toPlainText()
        if enter_text_description == "":
            self.terminal_message("Please enter text description", is_error=True)
            return
        # 获取检索 style
        search_style = self.comboBox_search_style.currentText()        
        if search_style == "Static Search":
            self.terminal_message("Search style: Static Search")            
            self.terminal_message("Query:")
            self.terminal_message(enter_text_description)
            # 检查静态数据库及图像索引是否完备
            if self.check_static_database():
                result_sim, result_pids, result_image_paths, dataset_base_path = self.static_search(enter_text_description)                
                self.show_search_result(result_sim, result_image_paths, dataset_base_path)
            else:
                self.terminal_message("ERROR: Please check static database!", is_error=True)
                return
        elif search_style == "Dynamic Search":
            self.terminal_message("Search style: Dynamic Search")            
            self.terminal_message("Query:")
            self.terminal_message(enter_text_description)            
            if self.get_dynamic_database():
                result_sim, result_image_ids, result_image_paths, dataset_base_path = self.dynamic_search(enter_text_description)
                self.show_search_result(result_sim, result_image_paths, dataset_base_path)
            else:
                self.terminal_message("ERROR: Dynamic data path dose not contain an image file!", is_error=True)
                return
        else:
            self.terminal_message("Please select search style", is_error=True)
            return

    def slot_save_dataset(self):
        save_npy_name = self.lineEdit_dynamic_to_static_name.text()
        if save_npy_name.endswith('.npy') is False:
            self.terminal_message("Please enter a valid file name, such as '*.npy'.")
            return
        # 保存动态图像特征
        save_feature_path = os.path.join(self.dynamic_database_base_path, save_npy_name)        
        np.save(save_feature_path, self.dynamic_image_features)
        # 保存图像对应路径
        save_image_path = save_feature_path.replace('.npy', '.json')
        json_data = {"img_paths": self.dynamic_database_image_files}
        json.dump(json_data, open(save_image_path, 'w'), indent=4)
        self.terminal_message("Save dynamic dataset successfully!")

    def slot_clean_terminal_output(self):
        self.textBrowser_terminal_output.clear()

    # ************************ deploy functions ************************ #
    def static_search(self, query_text):
        # 1.读取静态数据库
        test_image_norm_features = np.load(self.static_database_file_path)
        N = test_image_norm_features.shape[0]        
        with open(self.static_database_json_file_path, 'r') as f:
            static_database_json = json.load(f)
        # 获取数据集 base 目录
        dataset_base_path = os.path.dirname(self.static_database_file_path)
        self.update_progress_bar(1, 5)
        # # 2.获取文本特征
        # text = tokenize(query_text, tokenizer=self.tokenizer, text_length=77, truncate=True)
        # text = text.reshape((1, 77))
        # result = self.text_encoder.text_forward(text) # npu 计算     
        # text_feature = result[text.argmax(axis=-1), :] # 获取最大值的索引对应的特征，即为文本的 cls 特征        
        self.update_progress_bar(2, 5)
        # # 3.计算图像数据库特征与文本特征的相似度
        # similarity, index = [], []
        # loops = N // 1024
        # for i in range(loops):
        #     # 准备图像数据
        #     start_index = i * 1024 
        #     end_index = min((i + 1) * 1024, N)
        #     images = test_image_norm_features[start_index:end_index]
        #     # DEBUG 文本数据
        #     # text_feature = images[0, :]
        #     # 准备start_index数据
        #     start_index = np.array([start_index], dtype=np.int64) 
        #     inputs = [images, text_feature, start_index]
        #     result = self.consine_sim_model.similarity_forward(inputs) # npu 计算  
        #     similarity.append(result[0])
        #     index.append(result[1])        
        # # 处理不整除的情况
        # if N % 1024 != 0:
        #     start_index = loops * 1024
        #     images = np.zeros((1024, 512), dtype=np.float32)
        #     images[0 : N - start_index] = test_image_norm_features[start_index:]
        #     start_index = np.array([start_index], dtype=np.int64)
        #     inputs = [images, text_feature, start_index]
        #     result = self.consine_sim_model.similarity_forward(inputs)
        #     similarity.append(result[0])
        #     index.append(result[1])
        self.update_progress_bar(3, 5)
        # # 4.合并结果,并进行最终 TopK 操作    
        # similarity = np.concatenate(similarity, axis=1)
        # index = np.concatenate(index, axis=1)    
        # # 获取前 K 个最大值的索引
        # K = 10
        # sorted_indices = np.argsort(similarity, axis=1)[:, ::-1]
        # indices = sorted_indices[:, :K]
        # top10_values = np.take_along_axis(similarity, indices, axis=1).flatten()
        # top10_indices = np.take_along_axis(index, indices, axis=1).flatten()
        self.update_progress_bar(4, 5)
        # 5. 返回 Top10 的相似度值和对应的图像路径
        # DEBUG for development on not atlas200
        top10_values = np.random.rand(1, 10).flatten()
        top10_indices = np.random.randint(0, N, (1, 10)).flatten()   

        show_images_path =  [static_database_json['img_paths'][i] for i in top10_indices]
        self.update_progress_bar(5, 5)
        return top10_values, top10_indices, show_images_path, dataset_base_path

    def dynamic_search(self, query_text):

        database_image_files = self.dynamic_database_image_files
        dataset_base_path = self.dynamic_database_base_path
        total_bar = len(database_image_files) + 10     

        # # 1.获取文本特征
        # text = tokenize(query_text, tokenizer=self.tokenizer, text_length=77, truncate=True)
        # text = text.reshape((1, 77))
        # result = self.text_encoder.text_forward(text) # npu 计算     
        # text_feature = result[text.argmax(axis=-1), :] # 获取最大值的索引对应的特征，即为文本的 cls 特征 
        self.update_progress_bar(5, total_bar)
        # # 2.获取图像特征
        # image_features = []
        # i = 1
        # for image_file in database_image_files:
        #     img_path = os.path.join(dataset_base_path, image_file)
        #     om_input_image = transfer_pic(img_path)
        #     result = self.image_encoder.image_forward(om_input_image)
        #     # 归一化 om 模型推理结果
        #     om_image_feat = result[0, :].reshape(1, -1)
        #     om_image_feat = om_image_feat / np.linalg.norm(om_image_feat, ord=2, axis=-1, keepdims=True)
        #     image_features.append(om_image_feat) 
        #     i = i + 1
        #     self.update_progress_bar(5 + i, total_bar)
        # self.dynamic_image_features = np.concatenate(image_features, axis=0)                        
        # N = self.dynamic_image_features.shape[0]        
        # # 3.计算图像数据库特征与文本特征的相似度
        # similarity, index = [], []
        # loops = N // 1024
        # for i in range(loops):
        #     # 准备图像数据
        #     start_index = i * 1024 
        #     end_index = min((i + 1) * 1024, N)
        #     images = self.dynamic_image_features[start_index:end_index]
        #     # DEBUG 文本数据
        #     # text_feature = images[0, :]
        #     # 准备start_index数据
        #     start_index = np.array([start_index], dtype=np.int64) 
        #     inputs = [images, text_feature, start_index]
        #     result = self.consine_sim_model.similarity_forward(inputs) # npu 计算  
        #     similarity.append(result[0])
        #     index.append(result[1])        
        # # 处理不整除的情况
        # if N % 1024 != 0:
        #     start_index = loops * 1024
        #     images = np.zeros((1024, 512), dtype=np.float32)
        #     images[0 : N - start_index] = self.dynamic_image_features[start_index:]
        #     start_index = np.array([start_index], dtype=np.int64)
        #     inputs = [images, text_feature, start_index]
        #     result = self.consine_sim_model.similarity_forward(inputs)
        #     similarity.append(result[0])
        #     index.append(result[1])
        self.update_progress_bar(total_bar - 3, total_bar)
        # # 4.合并结果,并进行最终 TopK 操作    
        # similarity = np.concatenate(similarity, axis=1)
        # index = np.concatenate(index, axis=1)    
        # # 获取前 K 个最大值的索引
        # K = 10
        # sorted_indices = np.argsort(similarity, axis=1)[:, ::-1]
        # indices = sorted_indices[:, :K]
        # top10_values = np.take_along_axis(similarity, indices, axis=1).flatten()
        # top10_indices = np.take_along_axis(index, indices, axis=1).flatten()
        
        # 5. 返回 Top10 的相似度值和对应的图像路径
        # DEBUG for development on not atlas200
        self.dynamic_image_features = np.random.randn(500, 512)
        N = self.dynamic_image_features.shape[0]
        top10_values = np.random.rand(1, 10).flatten()
        top10_indices = np.random.randint(0, N, (1, 10)).flatten()  

        show_images_path =  [os.path.join(dataset_base_path, database_image_files[i]) for i in top10_indices]
        # 6. 设置保存动态图像特征文件名称
        self.lineEdit_dynamic_to_static_name.setText(f"{self.dynamic_dataset_folder_name}_test_data.npy")
        self.update_progress_bar(total_bar, total_bar)
        return top10_values, top10_indices, show_images_path, dataset_base_path

    # ************************ utils functions ************************ #
    def terminal_message(self, text, is_error=False):
        if is_error:
            self.textBrowser_terminal_output.append(f"<span style='color:red;'>{text}</span>")
        else:
            self.textBrowser_terminal_output.append(f"<span style='color:black;'>{text}</span>")
        self.textBrowser_terminal_output.moveCursor(self.textBrowser_terminal_output.textCursor().End)

    def check_static_database(self):
        static_database_file_path = self.lineEdit_select_dataset.text()
        if static_database_file_path is None:
            # 提示选择数据集
            self.terminal_message("Please select dataset", is_error=True)
            return False
        if static_database_file_path.lower().endswith('.npy') is False:
            # 提示选择.npy文件
            self.terminal_message("Please select '*.npy' file", is_error=True)
            return False         
        static_database_json_file_path = static_database_file_path.replace('.npy', '.json')
        if os.path.exists(static_database_json_file_path) is False:
            # 提示生成json文件
            self.terminal_message("Please generate json file for dataset", is_error=True)
            return False
        # 设置静态检索相关变量
        self.static_database_file_path = static_database_file_path
        self.static_database_json_file_path = static_database_json_file_path
        return True

    def get_dynamic_database(self):
        dynamic_database_path = self.lineEdit_select_path.text()
        if os.path.exists(dynamic_database_path) is False and os.path.isdir(dynamic_database_path) is False:
            # 提示选择数据集
            self.terminal_message("Please select true and exit data path", is_error=True)
            return False
        # 获取目录下的所有图像文件        
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')        
        image_files = []
        basepath = os.path.basename(dynamic_database_path) # 提取最后一级目录，制作图像相对路径
        self.dynamic_dataset_folder_name = basepath
        for f in os.listdir(dynamic_database_path):
            if f.lower().endswith(image_extensions):
                image_files.append(os.path.join(basepath, f))
        # 检查是否有图像文件
        if len(image_files) != 0:
            # 设置动态检索相关变量
            # self.dynamic_database_base_path + self.dynamic_database_image_files[i] 可以获取图像完成路径
            self.dynamic_database_base_path = os.path.dirname(dynamic_database_path)
            self.dynamic_database_image_files = image_files
            return True
        return False

    def show_search_result(self, result_sim, result_image_paths, dataset_base_path):
        for i in range(10):
            image_path = os.path.join(dataset_base_path, result_image_paths[i])
            sim = result_sim[i] 
            pixmap = QPixmap(image_path)
            resized_pixmap = pixmap.scaled(100, 200) 
            self.show_images_label_list[i].setPixmap(resized_pixmap)
            self.show_sim_label_list[i].setText(f"similarity: {sim:.3f}")
            self.show_images_label_list[i].setScaledContents(True)
            self.show_images_label_list[i].setAlignment(QtCore.Qt.AlignCenter)
            self.show_sim_label_list[i].setAlignment(QtCore.Qt.AlignCenter)
            
    def update_progress_bar(self, i, N):
        value = int(i / N * 100)
        # 更新进度条
        self.progressBar.setValue(value)
        
