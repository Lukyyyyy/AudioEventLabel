# -*- coding: utf-8 -*-
import sys
import os
import json
import ctypes
import traceback
import threading
import numpy as np
import sounddevice as sd
import colorsys
import subprocess
import csv
import shutil
from send2trash import send2trash

# ------------------------ 禁止子进程弹出黑框的代码 ------------------------
if sys.platform.startswith('win'):
    _original_popen = subprocess.Popen


    def _no_window_popen(*args, **kwargs):
        if 'startupinfo' not in kwargs:
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            kwargs['startupinfo'] = startupinfo

        if 'creationflags' not in kwargs:
            kwargs['creationflags'] = 0x08000000

        return _original_popen(*args, **kwargs)


    subprocess.Popen = _no_window_popen
# ---------------------------------------------------------------------

from pydub import AudioSegment
from PyQt5.QtWidgets import (QApplication, QMainWindow, QListWidget, QPushButton, QVBoxLayout, QWidget, QLabel,
                             QFileDialog, QMessageBox, QSlider, QSplitter, QMenuBar, QMenu, QComboBox,
                             QListWidgetItem, QAction, QAbstractItemView, QHBoxLayout)
from PyQt5.QtCore import Qt, pyqtSlot, QUrl
from PyQt5.QtGui import QIcon
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
import pyqtgraph as pg


class AudioLabelTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AudioEventLabel")

        # 获取程序所在的绝对路径
        if getattr(sys, 'frozen', False):
            self.base_path = os.path.dirname(sys.executable)
            self.resource_path = sys._MEIPASS
        else:
            self.base_path = os.path.dirname(os.path.abspath(__file__))
            self.resource_path = self.base_path

        self.config_path = os.path.join(self.base_path, 'config.json')

        # 设置 ffmpeg/ffprobe 路径
        ffmpeg_exe = os.path.join(self.resource_path, "ffmpeg.exe")
        ffprobe_exe = os.path.join(self.resource_path, "ffprobe.exe")
        if os.path.exists(ffmpeg_exe):
            AudioSegment.converter = ffmpeg_exe
            AudioSegment.ffprobe = ffprobe_exe

        # 默认配置
        self.config = {
            "annotation_dir": os.path.join(self.base_path, "annotations"),
            "categories_path": os.path.join(self.base_path, "categories.json")
        }
        self.load_config()

        # --- 定义目录和TSV路径 ---
        self.root_annotation_dir = self.config["annotation_dir"]

        # 定义 overlap 和 non-overlap 的文件夹路径
        self.overlap_dir = os.path.join(self.root_annotation_dir, "overlap")
        self.non_overlap_dir = os.path.join(self.root_annotation_dir, "non-overlap")

        # 定义对应的 TSV 路径
        self.overlap_tsv_path = os.path.join(self.overlap_dir, "overlap.tsv")
        self.non_overlap_tsv_path = os.path.join(self.non_overlap_dir, "non-overlap.tsv")

        # 初始化文件夹和TSV头
        self.init_directories_and_tsvs()

        # 缓存已处理的文件名
        self.processed_files_cache = set()
        self.refresh_processed_cache()

        # 核心数据变量
        self.current_audio_path = None
        self.y = None
        self.sr = None
        self.annotations_markers = []
        self.player = QMediaPlayer()

        # UI 初始化
        self.init_ui()
        self.init_plot()
        self.bind_signals()

        # 设置图标
        icon_file = os.path.join(self.resource_path, 'icon.png')
        if os.path.exists(icon_file):
            self.setWindowIcon(QIcon(icon_file))

        self.check_and_load_categories()
        self.set_windows_dark_titlebar()

    def load_config(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    saved_config = json.load(f)
                    self.config.update(saved_config)
            except Exception as e:
                print(f"读取配置失败: {e}")
        else:
            self.save_config()

    def save_config(self):
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"保存配置失败: {e}")

    # --- TSV 和 目录初始化 Start ---
    def init_directories_and_tsvs(self):
        """确保文件夹存在，并且TSV文件有表头"""
        for dir_path, tsv_path in [
            (self.overlap_dir, self.overlap_tsv_path),
            (self.non_overlap_dir, self.non_overlap_tsv_path)
        ]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            if not os.path.exists(tsv_path):
                try:
                    with open(tsv_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f, delimiter='\t')
                        writer.writerow(['filename', 'onset', 'offset', 'event_label'])
                except Exception as e:
                    self.show_error(f"创建TSV文件失败: {tsv_path}\n{e}")

    def refresh_processed_cache(self):
        """扫描 overlap 和 non-overlap 文件夹中的音频文件"""
        self.processed_files_cache.clear()

        # 扫描 overlap 文件夹
        if os.path.exists(self.overlap_dir):
            for f in os.listdir(self.overlap_dir):
                if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
                    self.processed_files_cache.add(f)

        # 扫描 non-overlap 文件夹
        if os.path.exists(self.non_overlap_dir):
            for f in os.listdir(self.non_overlap_dir):
                if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
                    self.processed_files_cache.add(f)

    def get_annotations_for_file(self, filename):
        """从两个TSV中查找该文件的标注"""
        annotations = []

        def read_tsv(path, target_name):
            res = []
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f, delimiter='\t')
                        for row in reader:
                            if row['filename'] == target_name:
                                # 只有当 event_label 不为空时才解析为有效标注
                                # 这样可以兼容只有 filename 的“无事件”记录
                                if row.get('event_label', '').strip():
                                    res.append({
                                        'start': float(row['onset']),
                                        'end': float(row['offset']),
                                        'category': row['event_label']
                                    })
                except:
                    pass
            return res

        annotations.extend(read_tsv(self.overlap_tsv_path, filename))
        annotations.extend(read_tsv(self.non_overlap_tsv_path, filename))

        return annotations

    def update_tsv_file(self, tsv_path, filename, new_annotations, keep_record_if_empty=False):
        """
        如果 new_annotations 为空，且 keep_record_if_empty=True，
        则写入一行只有 filename 但其他字段为空的记录。
        """
        all_rows = []
        header = ['filename', 'onset', 'offset', 'event_label']

        if os.path.exists(tsv_path):
            with open(tsv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                if reader.fieldnames:
                    header = reader.fieldnames
                for row in reader:
                    # 剔除当前文件的旧记录
                    if row['filename'] != filename:
                        all_rows.append(row)

        # 追加新数据
        if len(new_annotations) == 0 and keep_record_if_empty:
            # [新功能] 写入无事件记录：只有文件名，时间戳和标签留空
            all_rows.append({
                'filename': filename,
                'onset': '',
                'offset': '',
                'event_label': ''
            })
        else:
            # 正常写入标注
            for ann in new_annotations:
                all_rows.append({
                    'filename': filename,
                    'onset': f"{ann['start']:.3f}",
                    'offset': f"{ann['end']:.3f}",
                    'event_label': ann['category']
                })

        try:
            with open(tsv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=header, delimiter='\t')
                writer.writeheader()
                writer.writerows(all_rows)
        except Exception as e:
            self.show_error(f"写入TSV失败 ({os.path.basename(tsv_path)}): {e}")
            raise e

    def remove_file_from_all_tsvs(self, filename):
        """从所有TSV中移除记录（彻底删除）"""
        # 这里的 keep_record_if_empty 默认为 False，即彻底移除
        self.update_tsv_file(self.overlap_tsv_path, filename, [])
        self.update_tsv_file(self.non_overlap_tsv_path, filename, [])

    # --- TSV 相关辅助函数 End ---
    def check_and_load_categories(self):
        cat_path = self.config.get("categories_path", "")
        if not os.path.exists(cat_path):
            choice = QMessageBox.warning(
                self, "配置文件缺失",
                f"未能加载类别定义文件。\n路径不存在：\n{cat_path}\n\n请选择一个包含类别列表的 JSON 文件。",
                QMessageBox.Ok | QMessageBox.Cancel
            )
            if choice == QMessageBox.Ok:
                self.change_categories_path()
            else:
                self.category_combo_box.addItem("Default")
        else:
            self.load_categories_from_file(cat_path)

    def load_categories_from_file(self, path):
        self.category_combo_box.clear()
        try:
            with open(path, 'r', encoding='utf-8') as f:
                items = json.load(f)
                if items and isinstance(items, list):
                    self.category_combo_box.addItems(items)
                    return True
                else:
                    self.category_combo_box.addItem("Default")
                    self.show_error("类别文件格式错误：必须是 JSON 字符串列表。")
                    return False
        except Exception as e:
            self.show_error(f"读取类别文件失败: {e}")
            self.category_combo_box.addItem("Default")

    def init_ui(self):
        self.apply_modern_style()

        menubar = self.menuBar()
        file_menu = menubar.addMenu("文件")
        file_menu.addAction("打开文件", self.select_files)
        file_menu.addAction("打开文件夹", self.select_folder)
        settings_menu = menubar.addMenu("设置")
        settings_menu.addAction("设置标注保存目录", self.change_annotation_dir)
        settings_menu.addAction("导入类别文件", self.change_categories_path)

        # 左侧列表
        self.unprocessed_list = self.create_list_widget()
        self.processed_list = self.create_list_widget()
        self.audio_count_label = QLabel("0 个文件")
        self.audio_count_label.setStyleSheet("color: #888; margin-top: 5px;")

        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(8)

        l1 = QLabel("待处理音频")
        l1.setStyleSheet("font-weight: bold; font-size: 11pt; color: #4a90e2;")
        left_layout.addWidget(l1)
        left_layout.addWidget(self.unprocessed_list)

        l2 = QLabel("已处理音频")
        l2.setStyleSheet("font-weight: bold; font-size: 11pt; color: #66bb6a;")
        left_layout.addWidget(l2)
        left_layout.addWidget(self.processed_list)
        left_layout.addWidget(self.audio_count_label)

        left_widget = QWidget()
        left_widget.setLayout(left_layout)

        # 右侧工作区
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(15, 15, 15, 15)
        right_layout.setSpacing(15)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(True, True, alpha=0.3)
        self.plot_widget.setYRange(-1.1, 1.1)
        self.plot_widget.getPlotItem().getViewBox().setMouseEnabled(x=False, y=False)
        right_layout.addWidget(self.plot_widget, stretch=2)

        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(0, 0, 0, 0)

        self.start_slider = self.create_slider(self.update_range)
        self.end_slider = self.create_slider(self.update_range)
        self.start_time_label = QLabel("0.000s")
        self.end_time_label = QLabel("0.000s")

        control_layout.addLayout(self.create_slider_layout("起始点:", self.start_slider, self.start_time_label))
        control_layout.addLayout(self.create_slider_layout("结束点:", self.end_slider, self.end_time_label))

        btn_layout = QHBoxLayout()
        self.play_button = QPushButton("播放全段")
        self.play_button.setCursor(Qt.PointingHandCursor)
        self.play_button.setMinimumHeight(35)
        self.play_selected_button = QPushButton("仅播放选中 (红色区域)")
        self.play_selected_button.setCursor(Qt.PointingHandCursor)
        self.play_selected_button.setMinimumHeight(35)

        btn_layout.addWidget(self.play_button)
        btn_layout.addWidget(self.play_selected_button)
        control_layout.addLayout(btn_layout)
        right_layout.addWidget(control_panel)

        action_group = QWidget()
        action_group.setStyleSheet("background-color: #333; border-radius: 6px;")
        action_layout = QVBoxLayout(action_group)
        action_layout.setContentsMargins(10, 10, 10, 10)

        row1 = QHBoxLayout()
        self.category_combo_box = QComboBox()
        self.category_combo_box.setMinimumHeight(30)
        self.add_button = QPushButton("添加标注")
        self.add_button.setMinimumHeight(30)
        self.add_button.setStyleSheet("background-color: #d84315; border: none;")

        row1.addWidget(QLabel("当前类别:"))
        row1.addWidget(self.category_combo_box, 1)
        row1.addWidget(self.add_button)
        action_layout.addLayout(row1)

        self.annotation_list = QListWidget()
        self.annotation_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.annotation_list.customContextMenuRequested.connect(self.show_annotation_context_menu)
        self.annotation_list.setMaximumHeight(100)
        action_layout.addWidget(QLabel("已添加的事件片段:"))
        action_layout.addWidget(self.annotation_list)

        row_save = QHBoxLayout()
        self.save_button = QPushButton("保存")
        self.save_button.setMinimumHeight(40)
        self.save_button.setCursor(Qt.PointingHandCursor)
        self.save_button.setStyleSheet("""
            QPushButton { font-weight: bold; font-size: 11pt; background-color: #2e7d32; border-radius: 4px; }
            QPushButton:hover { background-color: #388e3c; }
        """)

        self.path_status_label = QLabel(f"保存目录: {self.root_annotation_dir}")
        self.path_status_label.setStyleSheet("color: gray; font-size: 12px;")

        row_save.addWidget(self.save_button)
        action_layout.addLayout(row_save)
        action_layout.addWidget(self.path_status_label, 0, Qt.AlignRight)

        right_layout.addWidget(action_group)

        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setHandleWidth(2)
        splitter.setSizes([280, 1020])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.resize(1300, 800)
        self.setCentralWidget(splitter)

    def set_windows_dark_titlebar(self):
        try:
            if sys.platform.startswith('win'):
                DWMWA_USE_IMMERSIVE_DARK_MODE = 20
                set_window_attribute = ctypes.windll.dwmapi.DwmSetWindowAttribute
                hwnd = int(self.winId())
                rendering_policy = ctypes.c_int(1)
                set_window_attribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, ctypes.byref(rendering_policy),
                                     ctypes.sizeof(rendering_policy))
                self.repaint()
        except:
            pass

    def change_annotation_dir(self):
        current_dir = self.config.get("annotation_dir", "")
        new_dir = QFileDialog.getExistingDirectory(self, "选择标注文件存放根目录", current_dir)

        if new_dir:
            self.config["annotation_dir"] = new_dir
            self.root_annotation_dir = new_dir
            self.overlap_dir = os.path.join(new_dir, "overlap")
            self.non_overlap_dir = os.path.join(new_dir, "non-overlap")
            self.overlap_tsv_path = os.path.join(self.overlap_dir, "overlap.tsv")
            self.non_overlap_tsv_path = os.path.join(self.non_overlap_dir, "non-overlap.tsv")

            self.save_config()
            self.init_directories_and_tsvs()

            self.path_status_label.setText(f"根目录: .../{os.path.basename(new_dir)}")

            self.refresh_processed_cache()
            self.refresh_file_lists_status()

            QMessageBox.information(self, "设置更新", "标注保存路径已更新，目录结构已初始化。")

    def change_categories_path(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择类别配置文件", self.base_path, "JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            self.config["categories_path"] = file_path
            self.save_config()
            if self.load_categories_from_file(file_path):
                QMessageBox.information(self, "成功", f"已加载类别文件：\n{os.path.basename(file_path)}")

    def refresh_file_lists_status(self):
        all_items = []
        for i in range(self.unprocessed_list.count()):
            all_items.append(self.unprocessed_list.item(i).data(Qt.UserRole))
        for i in range(self.processed_list.count()):
            all_items.append(self.processed_list.item(i).data(Qt.UserRole))

        self.unprocessed_list.clear()
        self.processed_list.clear()
        self.reset_ui_state()

        for path in all_items:
            self.add_file_to_list(path)

    def init_plot(self):
        self.main_curve = None
        self.plot_widget.setBackground('#1e1e1e')
        self.plot_widget.getAxis('left').setPen('#666')
        self.plot_widget.getAxis('left').setTextPen('#aaa')
        self.plot_widget.getAxis('bottom').setPen('#666')
        self.plot_widget.getAxis('bottom').setTextPen('#aaa')

    def create_list_widget(self):
        lw = QListWidget()
        lw.setContextMenuPolicy(Qt.CustomContextMenu)
        lw.customContextMenuRequested.connect(self.show_file_list_context_menu)
        return lw

    def create_slider(self, callback):
        slider = QSlider(Qt.Horizontal)
        slider.valueChanged.connect(callback)
        return slider

    def create_slider_layout(self, label_text, slider, time_label):
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label_text))
        layout.addWidget(slider)
        layout.addWidget(time_label)
        return layout

    def bind_signals(self):
        self.unprocessed_list.itemClicked.connect(self.on_file_selected)
        self.processed_list.itemClicked.connect(self.on_file_selected)
        self.play_button.clicked.connect(self.play_audio)
        self.play_selected_button.clicked.connect(self.play_selected_audio)
        self.add_button.clicked.connect(self.add_annotation)
        self.save_button.clicked.connect(self.save_annotations)

    @pyqtSlot()
    def on_file_selected(self):
        sender = self.sender()
        if isinstance(sender, QListWidget):
            target_list = sender
        else:
            target_list = self.unprocessed_list

        item = target_list.currentItem()
        if not item: return

        audio_path = item.data(Qt.UserRole)
        if self.current_audio_path == audio_path and self.y is not None:
            return

        self.reset_ui_state(keep_selection=True)
        self.current_audio_path = audio_path

        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(audio_path)))
        if self.load_waveform_data(audio_path):
            self.restore_annotations(audio_path)

    def load_waveform_data(self, audio_path):
        try:
            file_name = os.path.basename(audio_path)
            self.plot_widget.setTitle(f"{file_name}", color='#ffffff', size='12pt')

            audio = AudioSegment.from_file(audio_path)
            if audio.channels > 1: audio = audio.set_channels(1)

            data = np.array(audio.get_array_of_samples())
            self.y = data.astype(np.float32) / (2 ** (audio.sample_width * 8 - 1))
            self.sr = audio.frame_rate

            max_val = len(self.y) - 1
            self.start_slider.setMaximum(max_val)
            self.end_slider.setMaximum(max_val)
            self.start_slider.setValue(0)
            self.end_slider.setValue(max_val)
            self.update_waveform()
            return True
        except Exception as e:
            self.show_error(f"加载音频失败: {str(e)}")
            return False

    @pyqtSlot()
    def update_range(self):
        s_val = self.start_slider.value()
        e_val = self.end_slider.value()
        sender = self.sender()
        if sender == self.start_slider and s_val > e_val:
            self.end_slider.setValue(s_val)
        elif sender == self.end_slider and e_val < s_val:
            self.start_slider.setValue(e_val)

        if self.sr:
            self.start_time_label.setText(f"{self.start_slider.value() / self.sr:.3f}s")
            self.end_time_label.setText(f"{self.end_slider.value() / self.sr:.3f}s")
        self.update_waveform()

    def update_waveform(self):
        if self.y is None or self.sr is None: return
        start_idx = self.start_slider.value()
        end_idx = self.end_slider.value()

        self.plot_widget.clear()
        self.annotations_markers = []

        x_full = np.arange(len(self.y)) / self.sr
        self.plot_widget.plot(x_full, self.y, pen='w', alpha=0.5)

        if start_idx < end_idx:
            x_sel = np.arange(start_idx, end_idx) / self.sr
            y_sel = self.y[start_idx:end_idx]
            self.plot_widget.plot(x_sel, y_sel, pen='r')

        self.plot_widget.addItem(pg.InfiniteLine(pos=start_idx / self.sr, angle=90, pen='g', movable=False))
        self.plot_widget.addItem(pg.InfiniteLine(pos=end_idx / self.sr, angle=90, pen='g', movable=False))

        for i in range(self.annotation_list.count()):
            self.draw_annotation_on_plot(self.annotation_list.item(i).text())

    def draw_annotation_on_plot(self, text):
        try:
            time_part, cat_part = text.split(':')
            start_t, end_t = map(float, time_part.split('-'))
            category = cat_part.strip()

            cats = [self.category_combo_box.itemText(i) for i in range(self.category_combo_box.count())]
            color_idx = cats.index(category) if category in cats else 0
            rgb = colorsys.hsv_to_rgb(color_idx / max(1, len(cats)), 0.8, 0.8)
            color = (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255), 100)

            region = pg.LinearRegionItem([start_t, end_t], brush=color, movable=False)
            label = pg.TextItem(category, color='w', anchor=(0.5, 1))
            label.setPos((start_t + end_t) / 2, 1.05)

            self.plot_widget.addItem(region)
            self.plot_widget.addItem(label)
            self.annotations_markers.append({'region': region, 'label': label, 'text': text})
        except:
            pass

    @pyqtSlot()
    def add_annotation(self):
        if not self.sr: return
        start = self.start_slider.value() / self.sr
        end = self.end_slider.value() / self.sr
        cat = self.category_combo_box.currentText()
        txt = f"{start:.3f}-{end:.3f}: {cat}"
        self.annotation_list.addItem(txt)
        self.draw_annotation_on_plot(txt)

    def check_is_overlap(self, annotations):
        """判断标注列表是否存在时间重叠"""
        if len(annotations) < 2:
            return False
        sorted_anns = sorted(annotations, key=lambda x: x['start'])
        for i in range(1, len(sorted_anns)):
            prev = sorted_anns[i - 1]
            curr = sorted_anns[i]
            if curr['start'] < prev['end']:
                return True
        return False

    @pyqtSlot()
    def save_annotations(self):
        if not self.current_audio_path: return

        # 1. 收集当前UI上的标注
        new_data = []
        for i in range(self.annotation_list.count()):
            txt = self.annotation_list.item(i).text()
            try:
                t, c = txt.split(':')
                s, e = map(float, t.split('-'))
                new_data.append({'start': s, 'end': e, 'category': c.strip()})
            except Exception as e:
                self.show_error(f"解析标注错误: {e}")
                return

        file_name = os.path.basename(self.current_audio_path)
        source_path = self.current_audio_path

        try:
            is_empty_labels = (len(new_data) == 0)
            is_overlap = self.check_is_overlap(new_data)

            target_operations = []

            if is_empty_labels:
                target_operations = [
                    (self.overlap_tsv_path, self.overlap_dir, 'copy'),
                    (self.non_overlap_tsv_path, self.non_overlap_dir, 'move')
                ]
            else:
                if is_overlap:
                    target_operations = [(self.overlap_tsv_path, self.overlap_dir, 'move')]
                else:
                    target_operations = [(self.non_overlap_tsv_path, self.non_overlap_dir, 'move')]

            final_dest_path = None

            for tsv, target_dir, method in target_operations:
                # [修改] 传递 keep_record_if_empty 参数
                # 如果是无事件的情况，我们需要在TSV中强制写入一条只有文件名的记录
                self.update_tsv_file(tsv, file_name, new_data, keep_record_if_empty=is_empty_labels)

                dest_path = os.path.join(target_dir, file_name)

                if os.path.normpath(source_path) != os.path.normpath(dest_path):
                    if os.path.exists(dest_path):
                        os.remove(dest_path)

                    if method == 'copy':
                        shutil.copy2(source_path, dest_path)
                    elif method == 'move':
                        shutil.move(source_path, dest_path)
                        self.current_audio_path = dest_path
                        final_dest_path = dest_path

            if final_dest_path is None and os.path.exists(source_path):
                final_dest_path = source_path

            # 清理旧数据逻辑 (这里的 update_tsv_file 默认 keep=False，所以会删除记录)
            if not is_empty_labels:
                if is_overlap:
                    self.update_tsv_file(self.non_overlap_tsv_path, file_name, [])
                else:
                    self.update_tsv_file(self.overlap_tsv_path, file_name, [])

            msg_suffix = " (重叠)" if is_overlap else " (非重叠)"
            if is_empty_labels: msg_suffix = " (无事件 - 双重归档)"

            QMessageBox.information(self, "成功", f"文件已处理{msg_suffix}\n并已移动至对应文件夹。")

            # 更新UI列表
            self.processed_files_cache.add(file_name)

            items_to_move = []

            # 在 Unprocessed 找
            search_matches = self.unprocessed_list.findItems(file_name, Qt.MatchExactly)
            for item in search_matches:
                items_to_move.append((self.unprocessed_list, item))

            # 在 Processed 找
            if not items_to_move:
                search_matches_p = self.processed_list.findItems(file_name, Qt.MatchExactly)
                for item in search_matches_p:
                    items_to_move.append((self.processed_list, item))

            for list_widget, item in items_to_move:
                row = list_widget.row(item)
                taken_item = list_widget.takeItem(row)
                if final_dest_path:
                    taken_item.setData(Qt.UserRole, final_dest_path)
                    taken_item.setToolTip(final_dest_path)
                self.processed_list.insertItem(0, taken_item)

            self.update_audio_count()
            self.auto_load_next()

        except Exception as e:
            self.show_error(f"保存处理失败: {traceback.format_exc()}")

    def auto_load_next(self):
        if self.unprocessed_list.count() > 0:
            self.unprocessed_list.setCurrentRow(0)
            self.on_file_selected()
        else:
            self.reset_ui_state()

    def show_file_list_context_menu(self, pos):
        sender_list = self.sender()
        item = sender_list.itemAt(pos)
        if item:
            menu = QMenu()
            menu.setStyleSheet("""
                QMenu {
                    background-color: #2b2b2b;
                    border: 1px solid #3e3e3e;
                    border-radius: 6px;
                    padding: 4px;
                }
                QMenu::item {
                    padding: 8px 25px;
                    color: #eee;
                    border-radius: 4px;
                    margin: 2px;
                }
                QMenu::item:selected {
                    background-color: #c62828;
                    color: white;
                    font-weight: bold;
                }
            """)
            if sender_list == self.processed_list:
                action = menu.addAction("🗑️ 删除文件(含重叠/非重叠)及记录")
            elif sender_list == self.unprocessed_list:
                action = menu.addAction("🗑️ 删除文件")
            action.triggered.connect(lambda: self.delete_file_item(sender_list, item))
            menu.exec_(sender_list.mapToGlobal(pos))

    def show_annotation_context_menu(self, pos):
        item = self.annotation_list.itemAt(pos)
        if item:
            menu = QMenu()
            menu.setStyleSheet("""
                QMenu {
                    background-color: #2b2b2b;
                    border: 1px solid #3e3e3e;
                    border-radius: 6px;
                    padding: 4px;
                }
                QMenu::item {
                    padding: 8px 25px;
                    color: #eee;
                    border-radius: 4px;
                    margin: 2px;
                }
                QMenu::item:selected {
                    background-color: #d84315;
                    color: white;
                    font-weight: bold;
                }
            """)
            action = menu.addAction("✕ 删除此标注")
            action.triggered.connect(lambda: self.delete_annotation_item(item))
            menu.exec_(self.annotation_list.mapToGlobal(pos))

    def delete_file_item(self, list_widget, item):
        raw_path = item.data(Qt.UserRole)
        path = os.path.normpath(raw_path)
        file_name = os.path.basename(path)

        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除文件 \"{file_name}\" 吗？\n\n该操作会尝试从 Overlap 和 Non-overlap 文件夹中查找并删除该文件及其标注记录。",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes: return

        row = list_widget.row(item)
        list_widget.takeItem(row)

        try:
            self.remove_file_from_all_tsvs(file_name)
        except Exception as e:
            self.show_error(f"更新TSV失败: {str(e)}")

        deleted_count = 0
        for d in [self.overlap_dir, self.non_overlap_dir]:
            p = os.path.join(d, file_name)
            if os.path.exists(p):
                try:
                    send2trash(p)
                    deleted_count += 1
                except Exception as e:
                    print(f"删除失败 {p}: {e}")

        if os.path.exists(path):
            try:
                send2trash(path)
                deleted_count += 1
            except:
                pass

        if deleted_count == 0:
            if os.path.exists(path):
                send2trash(path)

        self.update_audio_count()
        current_path_norm = os.path.normpath(self.current_audio_path) if self.current_audio_path else None

        if current_path_norm and file_name == os.path.basename(current_path_norm):
            self.reset_ui_state()
            self.plot_widget.setTitle("")

        QMessageBox.information(self, "操作成功", f"文件 \"{file_name}\" 已移除。")

    def delete_annotation_item(self, item):
        row = self.annotation_list.row(item)
        self.annotation_list.takeItem(row)
        self.update_waveform()

    def reset_ui_state(self, keep_selection=False):
        self.player.stop()
        self.y = None
        self.sr = None
        if not keep_selection:
            self.current_audio_path = None
        self.start_slider.setValue(0)
        self.end_slider.setValue(0)
        self.start_time_label.setText("0.000s")
        self.end_time_label.setText("0.000s")
        self.plot_widget.clear()
        self.annotation_list.clear()
        self.annotations_markers = []

    def select_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择音频", "", "Audio (*.wav *.mp3 *.flac *.ogg *.aac *.m4a *.wma *.aiff)"
        )
        if files:
            self.refresh_processed_cache()
            for f in files:
                self.add_file_to_list(f)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder:
            self.refresh_processed_cache()
            for f in os.listdir(folder):
                if f.endswith(('.wav', '.mp3', '.flac', '.ogg', '.aac', '.m4a', '.wma', '.aiff')):
                    self.add_file_to_list(os.path.join(folder, f))

    def add_file_to_list(self, path):
        file_name = os.path.basename(path)
        is_processed = file_name in self.processed_files_cache

        target_list = self.processed_list if is_processed else self.unprocessed_list

        exists = False
        for i in range(target_list.count()):
            if target_list.item(i).data(Qt.UserRole) == path:
                exists = True
                break

        if target_list == self.processed_list:
            for i in range(target_list.count()):
                if os.path.basename(target_list.item(i).data(Qt.UserRole)) == file_name:
                    exists = True
                    break

        if not exists:
            item = QListWidgetItem(file_name)
            item.setData(Qt.UserRole, path)
            item.setToolTip(path)
            target_list.addItem(item)
        self.update_audio_count()

    def update_audio_count(self):
        c1 = self.unprocessed_list.count()
        c2 = self.processed_list.count()
        self.audio_count_label.setText(f"{c1 + c2} 个文件 (未处理: {c1}, 已处理: {c2})")

    def restore_annotations(self, audio_path):
        self.annotation_list.clear()
        filename = os.path.basename(audio_path)
        data = self.get_annotations_for_file(filename)

        for item in data:
            txt = f"{item['start']:.3f}-{item['end']:.3f}: {item['category']}"
            self.annotation_list.addItem(txt)

        if data:
            self.update_waveform()

    def play_audio(self):
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def play_selected_audio(self):
        if self.y is not None and self.sr:
            s = self.start_slider.value()
            e = self.end_slider.value()
            if s < e:
                threading.Thread(target=lambda: sd.play(self.y[s:e], self.sr)).start()

    def show_error(self, msg):
        QMessageBox.critical(self, "错误", msg)

    def apply_modern_style(self):
        style_sheet = """
        QMainWindow, QWidget { background-color: #2b2b2b; color: #e0e0e0; font-family: "Segoe UI", sans-serif; font-size: 10pt; }
        QListWidget { background-color: #1e1e1e; border: 1px solid #3e3e3e; border-radius: 4px; outline: none; }
        QListWidget::item { padding: 8px; border-bottom: 1px solid #2e2e2e; }
        QListWidget::item:selected { background-color: #37373d; border-left: 3px solid #4a90e2; color: #ffffff; }
        QListWidget::item:hover { background-color: #2a2d35; }
        QPushButton { background-color: #3c3c3c; border: 1px solid #555; border-radius: 4px; padding: 6px 12px; color: #fff; }
        QPushButton:hover { background-color: #4a90e2; border: 1px solid #4a90e2; }
        QPushButton:pressed { background-color: #357abd; }
        QPushButton[text="保存并自动归类"] { background-color: #2e7d32; border: none; }
        QPushButton[text="保存并自动归类"]:hover { background-color: #388e3c; }
        QSlider::groove:horizontal { border: 1px solid #3d3d3d; height: 6px; background: #1e1e1e; margin: 2px 0; border-radius: 3px; }
        QSlider::handle:horizontal { background: #b0b0b0; border: 1px solid #5c5c5c; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }
        QSlider::handle:horizontal:hover { background: #4a90e2; }
        QComboBox { background-color: #1e1e1e; border: 1px solid #3e3e3e; border-radius: 4px; padding: 4px; }
        QComboBox::drop-down { border: none; }
        QComboBox QAbstractItemView { background-color: #1e1e1e; selection-background-color: #4a90e2; color: #e0e0e0; }
        QSplitter::handle { background-color: #3e3e3e; }
        QMenu { background-color: #2d2d2d; border: 1px solid #454545; border-radius: 6px; padding: 4px 0; }
        QMenu::item { background-color: transparent; padding: 6px 20px; margin: 2px 5px; color: #e0e0e0; border-radius: 4px; }
        QMenu::item:selected { background-color: #4a90e2; color: #ffffff; }
        QMenu::separator { height: 1px; background-color: #555; margin: 4px 0; }
        QMenuBar::item:selected { background-color: #5d5d5d; color: #ffffff; }
        """
        self.setStyleSheet(style_sheet)


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
    else:
        print("未捕获的异常:", exc_type, exc_value)
        traceback.print_exception(exc_type, exc_value, exc_traceback)


if __name__ == '__main__':
    sys.excepthook = handle_exception
    app = QApplication(sys.argv)
    window = AudioLabelTool()
    window.show()
    sys.exit(app.exec_())
