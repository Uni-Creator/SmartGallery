import os
import subprocess
from queue import Queue
import threading
from io import BytesIO

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QLineEdit, QVBoxLayout, QHBoxLayout,
    QGridLayout, QScrollArea, QListWidget,
    QListWidgetItem, QDialog
)

from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PIL import Image

import os
os.environ["QT_LOGGING_RULES"] = "qt.gui.icc=false"

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# ================= THUMBNAIL WORKER =================

class ThumbnailWorker(QObject):
    thumbnail_loaded = pyqtSignal(str, QPixmap)

    def __init__(self):
        super().__init__()
        self.queue = Queue()
        self.thread = threading.Thread(target=self.process_queue, daemon=True)
        self.thread.start()

    def add_task(self, path):
        self.queue.put(path)

    def process_queue(self):
        while True:
            path = self.queue.get()

            if path is None:
                break

            try:
                img = Image.open(path)
                img.thumbnail((200, 200))

                if img.mode != "RGB":
                    img = img.convert("RGB")

                buffer = BytesIO()
                img.save(buffer, format="PNG")
                buffer.seek(0)

                pixmap = QPixmap.fromImage(QImage.fromData(buffer.read()))
                self.thumbnail_loaded.emit(path, pixmap)
            except:
                pass

# ================= IMAGE VIEWER =================

class FullImageViewer(QDialog):
    def __init__(self, image_path):
        super().__init__()
        self.setWindowTitle(os.path.basename(image_path))
        self.resize(900, 700)

        layout = QVBoxLayout()
        self.label = QLabel("Loading...")
        self.label.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.label)
        self.setLayout(layout)

        self.image_path = image_path
        self.load_image()

    def load_image(self):
        pix = QPixmap(self.image_path)
        pix = pix.scaled(880, 650, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label.setPixmap(pix)

# ================= MAIN WINDOW =================

class GalleryWindow(QMainWindow):
    def __init__(self, base_folder):
        super().__init__()

        self.setWindowTitle("Smart Gallery")
        self.resize(1400, 800)

        self.base_folder = base_folder
        self.thumbnail_cache = {}
        self.thumbnail_labels = {}

        # Persistent thumbnail worker
        self.thumbnail_worker = ThumbnailWorker()
        self.thumbnail_worker.thumbnail_loaded.connect(self.on_thumbnail_loaded)

        # Persistent search server
        self.search_process = subprocess.Popen(
            ["python", "search_server.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True
        )

        while True:
            if self.search_process.stdout.readline().strip() == "READY":
                break

        self.album_map = self.build_album_map()
        self.current_album = "All Photos"
        self.current_images = self.album_map[self.current_album]

        self.batch_size = 100
        self.loaded_count = 0

        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        # Left album panel
        self.album_list = QListWidget()
        for album in self.album_map.keys():
            self.album_list.addItem(QListWidgetItem(album))
        self.album_list.currentItemChanged.connect(self.change_album)

        main_layout.addWidget(self.album_list, 1)

        # Right panel
        right_layout = QVBoxLayout()

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Type and press ENTER to search...")
        self.search_box.returnPressed.connect(self.run_search)

        right_layout.addWidget(self.search_box)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.verticalScrollBar().valueChanged.connect(self.on_scroll)

        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.scroll.setWidget(self.grid_widget)

        right_layout.addWidget(self.scroll)
        main_layout.addLayout(right_layout, 5)

        self.album_list.setCurrentRow(0)
        self.load_batch()

    # ================= ALBUM MANAGEMENT =================

    def build_album_map(self):
        album_map = {"All Photos": []}

        for root, _, files in os.walk(self.base_folder):
            for file in files:
                if file.lower().endswith(IMAGE_EXTS):
                    path = os.path.join(root, file)

                    album_map["All Photos"].append(path)

                    rel = os.path.relpath(root, self.base_folder)
                    album = rel.split(os.sep)[0] if rel != "." else "Misc"

                    album_map.setdefault(album, []).append(path)

        return album_map

    def change_album(self):
        album = self.album_list.currentItem().text()
        self.current_album = album
        self.current_images = self.album_map[album]

        self.loaded_count = 0
        self.clear_grid()

        self.filtered_images = self.current_images
        self.load_batch()

    # ================= SEARCH =================

    def run_search(self):
        query = self.search_box.text().strip()

        if query == "":
            self.filtered_images = self.current_images
        else:
            self.filtered_images = self.ai_search(query)

        self.loaded_count = 0
        self.clear_grid()
        self.load_batch()

    def ai_search(self, query):
        self.search_process.stdin.write(query + "\n")
        self.search_process.stdin.flush()

        results = []
        while True:
            line = self.search_process.stdout.readline().strip()
            if line == "END":
                break
            results.append(line)

        return results

    # ================= GRID HANDLING =================

    def load_batch(self):
        images = self.filtered_images
        end = min(self.loaded_count + self.batch_size, len(images))

        row = self.loaded_count // 4
        col = 0

        for i in range(self.loaded_count, end):
            path = images[i]

            label = QLabel("Loading...")
            self.thumbnail_labels[path] = label

            self.grid_layout.addWidget(label, row, col)
            self.thumbnail_worker.add_task(path)

            label.mousePressEvent = lambda e, p=path: self.open_image(p)

            col += 1
            if col >= 4:
                col = 0
                row += 1

        self.loaded_count = end

    def clear_grid(self):
        self.thumbnail_labels.clear()
        for i in reversed(range(self.grid_layout.count())):
            widget = self.grid_layout.itemAt(i).widget()
            widget.deleteLater()

    def on_thumbnail_loaded(self, path, pix):
        if path in self.thumbnail_labels:
            self.thumbnail_labels[path].setPixmap(pix)

    def on_scroll(self):
        scrollbar = self.scroll.verticalScrollBar()
        if scrollbar.value() > scrollbar.maximum() - 150:
            self.load_batch()

    # ================= VIEWER =================

    def open_image(self, path):
        viewer = FullImageViewer(path)
        viewer.exec_()

    # ================= CLEAN EXIT =================

    def closeEvent(self, event):
        try:
            self.search_process.stdin.write("EXIT\n")
            self.search_process.stdin.flush()
            self.search_process.kill()
        except:
            pass
        event.accept()

# ================= ENTRY =================

if __name__ == "__main__":
    app = QApplication([])

    BASE_FOLDER = r"D:\Personal\PHOTOS"
    window = GalleryWindow(BASE_FOLDER)

    window.show()
    app.exec_()
