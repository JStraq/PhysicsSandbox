# -*- coding: utf-8 -*-
"""
Provides base classes and configurations for running physics visualizations
Based in PyQt5 GUI framework with matplotlib
"""

import sys
import PyQt5.QtCore as qc
import PyQt5.QtGui as qg
import PyQt5.QtWidgets as qw
import matplotlib
matplotlib.use('Qt5Agg')

class window(qw.QMainWindow):
   def __init__(self, parent = None):
      super(window, self).__init__(parent)
      
      self.canvas = MplCanvas()
      self.canvas.axes.plot([0,1,2,3,4], [3,5,2,3,1])
      
      self.setCentralWidget(self.canvas)
      
      
class MplCanvas(matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = matplotlib.figure.Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


def main():
   app = qw.QApplication(sys.argv)
   ex = window()
   ex.show()
   sys.exit(app.exec_())


if __name__ == '__main__':
   main()