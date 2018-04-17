from PyQt5 import QtGui


def make_qimage(img):

    h, w = img.shape[:2]
    if img.ndim == 2:
        bytesPerLine = w
        img_format = QtGui.QImage.Format_Grayscale8
    else:
        bytesPerLine = 3 * w
        img_format = QtGui.QImage.Format_RGB888

    qimage = QtGui.QImage(img.data, w, h, bytesPerLine, img_format)

    return qimage
