# main.py
from PyQt6.QtWidgets import QApplication
from MainWindow import MainApp
import sys

def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
