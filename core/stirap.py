import sys

import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QPushButton
from functools import partial
from rydberg_awg import RydbergPulseAwg

sys.path.append("..")
from res.main_window import Ui_MainWindow
from res.sequence import Ui_sequence
from res.card_config import Ui_config


class STIRAP(QMainWindow, Ui_sequence):
    def __init__(self):
        super(STIRAP, self).__init__()
        self.setupUi(self)
