import sys
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QPushButton, QLabel, QLineEdit, QSpinBox, QTextEdit, 
    QProgressBar, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from kcppvideo import analyze_video

""" Generated entirely by Claude Sonnet 3.5 """

class VideoAnalysisWorker(QThread):
    """ Worker thread to run video analysis without blocking GUI. """
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, video_path, api_url, template_dir, max_frames):
        super().__init__()
        self.video_path = video_path
        self.api_url = api_url
        self.template_dir = template_dir
        self.max_frames = max_frames
        
    def run(self):
        try:
            self.progress.emit("Starting video analysis...")
            results = analyze_video(
                self.video_path,
                self.api_url,
                self.template_dir,
                self.max_frames
            )
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))

class VideoAnalyzerGUI(QMainWindow):
    """ Main window for video analysis application. """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Analysis Tool")
        self.setMinimumSize(800, 600)
        
        # Initialize member variables
        self.video_path = None
        self.worker = None
        
        self.init_ui()
        
    def init_ui(self):
        """ Set up the user interface. """
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # File selection
        file_layout = QHBoxLayout()
        self.file_label = QLabel("No video selected")
        select_btn = QPushButton("Select Video")
        select_btn.clicked.connect(self.select_video)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(select_btn)
        layout.addLayout(file_layout)
        
        # Settings
        settings_layout = QVBoxLayout()
        
        # API URL
        url_layout = QHBoxLayout()
        url_layout.addWidget(QLabel("API URL:"))
        self.url_input = QLineEdit("http://localhost:5001")
        url_layout.addWidget(self.url_input)
        settings_layout.addLayout(url_layout)
        
        # Password (if needed)
        pass_layout = QHBoxLayout()
        pass_layout.addWidget(QLabel("Password:"))
        self.pass_input = QLineEdit()
        self.pass_input.setEchoMode(QLineEdit.EchoMode.Password)
        pass_layout.addWidget(self.pass_input)
        settings_layout.addLayout(pass_layout)
        
        # Max frames
        frames_layout = QHBoxLayout()
        frames_layout.addWidget(QLabel("Max Frames:"))
        self.frames_input = QSpinBox()
        self.frames_input.setRange(2, 100)
        self.frames_input.setValue(24)
        frames_layout.addWidget(self.frames_input)
        settings_layout.addLayout(frames_layout)
        
        layout.addLayout(settings_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # Status/Log area
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        layout.addWidget(self.log_area)
        
        # Control buttons
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Analysis")
        self.start_btn.clicked.connect(self.start_analysis)
        self.start_btn.setEnabled(False)
        button_layout.addWidget(self.start_btn)
        
        layout.addLayout(button_layout)
        
    def select_video(self):
        """ Open file dialog to select video file. """
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)"
        )
        
        if file_name:
            self.video_path = file_name
            self.file_label.setText(Path(file_name).name)
            self.start_btn.setEnabled(True)
            self.log_message(f"Selected video: {file_name}")
    
    def start_analysis(self):
        """ Begin video analysis in separate thread. """
        if not self.video_path:
            QMessageBox.warning(self, "Error", "Please select a video first.")
            return
            
        self.start_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # Create and start worker thread
        self.worker = VideoAnalysisWorker(
            self.video_path,
            self.url_input.text(),
            "./templates",  # Could make this configurable
            self.frames_input.value()
        )
        
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.analysis_complete)
        self.worker.error.connect(self.handle_error)
        
        self.worker.start()
        
    def update_progress(self, message):
        """ Update progress bar and log. """
        self.log_message(message)
        # Could parse progress from message if needed
        self.progress_bar.setValue(self.progress_bar.value() + 10)
        
    def analysis_complete(self, results):
        """ Handle completed analysis. """
        self.progress_bar.setValue(100)
        self.start_btn.setEnabled(True)
        
        summary = results.get("final_summary", "No summary generated")
        self.log_message("\nAnalysis Complete!\n")
        self.log_message("Final Summary:")
        self.log_message("-" * 40)
        self.log_message(summary)
        
    def handle_error(self, error_msg):
        """ Handle analysis errors. """
        self.start_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        QMessageBox.critical(self, "Error", f"Analysis failed: {error_msg}")
        self.log_message(f"ERROR: {error_msg}")
        
    def log_message(self, message):
        """ Add message to log area. """
        self.log_area.append(message)

def main():
    app = QApplication(sys.argv)
    window = VideoAnalyzerGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()