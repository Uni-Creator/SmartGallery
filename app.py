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
import numpy as np
import hashlib
import json

os.environ["QT_LOGGING_RULES"] = "qt.gui.icc=false"

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

SEARCH_SERVER_CMD = ["python", "search_server.py"]
ENCODER_SERVER_CMD = ["python", "encoder_server.py"]
CACHE_THUMBNAILS_DIR = "data/.thumbnails"


# Step 1: Thumbnail background worker

class ThumbnailWorker(QObject):
    thumbnail_loaded = pyqtSignal(str, QImage)

    def __init__(self, cache, cache_dir=CACHE_THUMBNAILS_DIR):
        super().__init__()
        self.queue = Queue()
        self.cache = cache
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.index_path = os.path.join(self.cache_dir, "index.json")
        self._index = {}
        # Load index if exists (not the actual images yet)
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                self._index = json.load(f)
        except Exception as e:
            print("Thumbnail index load error:", e)
            self._index = {}

        self.thread = threading.Thread(target=self.process_queue, daemon=True)

    def add_task(self, path):
        # If thumbnail is already cached in memory, emit immediately
        if path in self.cache:
            self.thumbnail_loaded.emit(path, self.cache[path])
            return

        # If a thumbnail file exists on disk for this path, load it and emit
        h = hashlib.sha1(path.encode("utf-8")).hexdigest()
        filename = os.path.join(self.cache_dir, f"{h}.png")

        if os.path.exists(filename):
            image = QImage(filename)
            if not image.isNull():
                self.cache[path] = image
                self.thumbnail_loaded.emit(path, image)
                return

        # Otherwise queue for on-the-fly generation
        self.queue.put(path)

    def process_queue(self):
        while True:
            path = self.queue.get()

            if path is None:
                break

            try:
                image = QImage(path)

                if image.isNull():
                    continue

                image = image.scaled(
                    200,
                    200,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )

                # Cache QImage in memory
                self.cache[path] = image

                # Save to disk
                try:
                    h = hashlib.sha1(path.encode("utf-8")).hexdigest()
                    filename = os.path.join(self.cache_dir, f"{h}.png")
                    image.save(filename, "PNG")
                    self._index[h] = path
                    with open(self.index_path, "w", encoding="utf-8") as f:
                        json.dump(self._index, f)
                except Exception as e:
                    print("Thumbnail save error:", e)

                # Notify UI
                self.thumbnail_loaded.emit(path, image)

            except Exception as e:
                print("Thumbnail error:", e)

    def load_cache(self):
        # Load thumbnails from index and emit signals so the UI can update
        if not os.path.exists(self.index_path):
            # No persisted thumbnails yet
            return

        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                self._index = json.load(f)

            for h, path in self._index.items():
                filename = os.path.join(self.cache_dir, f"{h}.png")
                if os.path.exists(filename):
                    image = QImage(filename)
                    if not image.isNull():
                        self.cache[path] = image
                        # Emit so UI updates labels if present
                        self.thumbnail_loaded.emit(path, image)
        except Exception as e:
            print("Thumbnail load error:", e)



# Step 2: Full image viewer dialog

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

        self.load_image(image_path)

    def load_image(self, image_path):
        pix = QPixmap(image_path)
        pix = pix.scaled(880, 650, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label.setPixmap(pix)


# Step 3: Main gallery window

class GalleryWindow(QMainWindow):
    def __init__(self, base_folder):
        super().__init__()

        self.setWindowTitle("Smart Gallery")
        self.resize(1400, 800)

        self.base_folder = base_folder
        self.thumbnail_labels = {}

        # Step 3.1: Start search and encoder servers
        self.search_process = self.start_subprocess(SEARCH_SERVER_CMD, "READY")
        self.encoder_process = self.start_subprocess(ENCODER_SERVER_CMD, "ENCODER_READY")

        # Step 3.2: Build initial album map
        self.album_map = self.build_album_map()
        self.current_album = "All Photos"
        self.current_images = self.album_map[self.current_album]
        self.filtered_images = self.current_images

        self.batch_size = 100
        self.loaded_count = 0

        # Step 3.3: Initialize thumbnail worker
        self.thumbnail_cache = {}
        self.thumbnail_worker = ThumbnailWorker(self.thumbnail_cache)
        # Connect signal first so load_cache emits are received
        self.thumbnail_worker.thumbnail_loaded.connect(self.on_thumbnail_loaded)
        self.thumbnail_worker.load_cache()
        if self.thumbnail_cache:
            print("Thumbnail cache loaded.")
        else:
            print("No thumbnail cache found. Loading on-the-fly.")
            self.thumbnail_worker.thread.start()
            self.thumbnail_worker.save

        # Step 3.4: Build UI
        self.init_ui()
        self.load_batch()


    # Step 4: Subprocess startup helper
    def start_subprocess(self, command, ready_flag):
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1
        )

        while True:
            line = process.stdout.readline().strip()
            if line == ready_flag:
                break

        return process


    # Step 5: UI setup
    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        self.album_list = QListWidget()

        for album in self.album_map:
            self.album_list.addItem(album)

        self.album_list.currentItemChanged.connect(self.change_album)
        main_layout.addWidget(self.album_list, 1)

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


    # Step 6: Album structure management
    def build_album_map(self):
        album_map = {"All Photos": []}

        for root, _, files in os.walk(self.base_folder):
            for file in files:
                if file.lower().endswith(IMAGE_EXTS):
                    path = os.path.join(root, file)

                    album_map["All Photos"].append(path)

                    # Get relative path from base
                    rel = os.path.relpath(root, self.base_folder)

                    if rel == ".":
                        album = "Misc"
                    else:
                        parts = rel.split(os.sep)

                        # Build nested album name like: School / RKM
                        if len(parts) >= 2:
                            album = f"{parts[0]} / {parts[1]}"
                        else:
                            album = parts[0]

                    album_map.setdefault(album, []).append(path)

        return album_map



    def change_album(self):
        album = self.album_list.currentItem().text()

        self.current_album = album
        self.current_images = self.album_map.get(album, [])
        self.filtered_images = self.current_images

        self.loaded_count = 0
        self.clear_grid()
        self.load_batch()


    # Step 7: AI search handling
    def run_search(self):
        query = self.search_box.text().strip()

        if not query:
            self.filtered_images = self.current_images
        else:
            matched_images = self.ai_search(query)
            self.filtered_images = [
                path for path in self.current_images if path in matched_images
            ]
            

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


    # Step 8: Grid and infinite scroll handling
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

        while self.grid_layout.count():
            widget = self.grid_layout.takeAt(0).widget()
            if widget:
                widget.deleteLater()


    def on_thumbnail_loaded(self, path, image):
        if path in self.thumbnail_labels:

            pixmap = QPixmap.fromImage(image)
            self.thumbnail_labels[path].setPixmap(pixmap)



    def on_scroll(self):
        scrollbar = self.scroll.verticalScrollBar()

        if scrollbar.value() > scrollbar.maximum() - 150:
            self.load_batch()


    # Step 9: Image viewer
    def open_image(self, path):
        viewer = FullImageViewer(path)
        viewer.exec_()


    # Step 10: Real-time folder updates
    def on_directory_changed(self, dir_path):

        current_files = {
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if f.lower().endswith(IMAGE_EXTS)
        }

        known_files = set(self.album_map["All Photos"])

        new_files = current_files - known_files
        deleted_files = known_files - current_files

        for path in new_files:
            self.album_map["All Photos"].append(path)

            rel = os.path.relpath(dir_path, self.base_folder)

            if rel == ".":
                album = "Misc"
            else:
                parts = rel.split(os.sep)
                album = f"{parts[0]} / {parts[1]}" if len(parts) >= 2 else parts[0]


            self.album_map.setdefault(album, []).append(path)

            self.encoder_process.stdin.write(path + "\n")
            self.encoder_process.stdin.flush()

        for path in deleted_files:
            for album in self.album_map:
                if path in self.album_map[album]:
                    self.album_map[album].remove(path)

        self.current_images = self.album_map.get(self.current_album, [])
        self.filtered_images = self.current_images

        self.clear_grid()
        self.load_batch()


    # Step 11: Clean shutdown
    def closeEvent(self, event):
        try:
            self.encoder_process.stdin.write("EXIT\n")
            self.search_process.stdin.write("EXIT\n")

            self.encoder_process.stdin.flush()
            self.search_process.stdin.flush()

            self.encoder_process.terminate()
            self.search_process.terminate()

        except Exception as e:
            print("Shutdown error:", e)

        event.accept()


# Step 12: App entry point
if __name__ == "__main__":
    app = QApplication([])

    BASE_FOLDER = r"D:\Personal\PHOTOS"

    window = GalleryWindow(BASE_FOLDER)
    window.show()

    app.exec_()
