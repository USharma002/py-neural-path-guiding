from ui.app import MitsubaViewer
from PyQt6.QtWidgets import (
    QApplication
)

import sys
if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = MitsubaViewer()
    # viewer.showMaximized()
    viewer.show() 
    sys.exit(app.exec())