import sys
import pickle
from PyQt5 import QtWidgets, QtCore
from renderlayerdisplaywidget import RenderLayerDisplayWidget


class MainWindow(QtWidgets.QWidget):

    def __init__(self, path):
        super(MainWindow, self).__init__(parent=None)

        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, parent=self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(200)
        self.layout.addWidget(self.slider)

        results = pickle.load(open(path, "rb"))

        self.widgets = {}
        for render_name in results["results"].keys():
            img_original = results["results"][render_name]["img_original"]
            img_variant = results["results"][render_name]["img_variant"]
            dumps_by_alpha = results["results"][render_name]["by_alpha"]
            self.widgets[render_name] = RenderLayerDisplayWidget(img_original, img_variant, dumps_by_alpha, (32, 16), parent=self)
            self.layout.addWidget(self.widgets[render_name])

        self.slider.valueChanged.connect(self.slider_update)

    def slider_update(self, value):

        alpha = value / self.slider.maximum()

        for render_name in self.widgets.keys():
            self.widgets[render_name].set_alpha(alpha)


app = QtWidgets.QApplication(sys.argv)

path = sys.argv[1]
main_window = MainWindow(path)
main_window.show()
sys.exit(app.exec_())
