import numpy as np
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from conversion import make_qimage


class RenderLayerDisplayWidget(QtWidgets.QWidget):

    def __init__(self, img_original, img_variant, dumps_by_alpha, heatmap_shape, parent):
        super(RenderLayerDisplayWidget, self).__init__(parent)

        self.img_original = img_original
        self.img_variant = img_variant
        self.dumps_by_alpha = dumps_by_alpha
        self.heatmap_shape = heatmap_shape
        self.available_alphas = np.array(list(dumps_by_alpha.keys()))

        self.layout = QtWidgets.QHBoxLayout()
        self.setLayout(self.layout)

        self.label_img = QtWidgets.QLabel("No data", parent=self)
        self.label_dump = QtWidgets.QLabel("No data", parent=self)

        self.layout.addWidget(self.label_img)
        self.layout.addWidget(self.label_dump)

    def set_alpha(self, alpha):

        alpha_bot_index = np.argmin(np.clip(alpha - self.available_alphas, 0, None))
        alpha_top_index = min(alpha_bot_index + 1, len(self.available_alphas) - 1)

        alpha_bot = self.available_alphas[alpha_bot_index]
        alpha_top = self.available_alphas[alpha_top_index]

        if alpha_bot_index == alpha_top_index:
            d = 0
        else:
            d = (alpha - alpha_bot) / (alpha_top - alpha_bot)

        img_blended = cv2.addWeighted(self.img_original, 1.0 - alpha, self.img_variant, alpha, 0)

        dump_bot = self.dumps_by_alpha[alpha_bot]
        dump_top = self.dumps_by_alpha[alpha_top]

        dump_blended = (1 - d) * dump_bot + d * dump_top
        dump_reshaped = dump_blended.reshape(self.heatmap_shape)

        img_pixmap = QtGui.QPixmap(make_qimage(img_blended))
        self.label_img.setPixmap(img_pixmap)

        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        ax.imshow(dump_reshaped)
        canvas.draw()

        img_dump = np.fromstring(canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img_dump = img_dump.reshape(canvas.get_width_height()[::-1] + (3,))

        img_pixmap = QtGui.QPixmap(make_qimage(img_dump))
        self.label_dump.setPixmap(img_pixmap)
