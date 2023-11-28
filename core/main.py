#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
rydberg pulse controller

Author: yuezp
Last edited: 2022.11
"""

import sys

from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QPushButton, QLineEdit
from PyQt5.QtNetwork import QTcpServer, QTcpSocket, QHostAddress
from functools import partial
from rydberg_awg_test import RydbergPulseAwg
# from sequence import STIRAP, LandauZener, RabiFlopping, Sequence
from sequence_test import STIRAP, LandauZener, RabiFlopping, Sequence
from PyQt5.QtCore import Qt

from res.main_window import Ui_MainWindow


class Main(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(Main, self).__init__()
        self.setupUi(self)

        self.awg = RydbergPulseAwg()
        self.sequence = Sequence(self.awg, "SEQUENCE")
        self.stirap = Sequence(self.awg, "STIRAP")
        self.lz = Sequence(self.awg, "LZ")
        self.rabi = Sequence(self.awg, "RABI")

        self.tcpserver = QTcpServer()
        self.tcpsocket = QTcpSocket()

        self.tcpserver.listen(QHostAddress.Any, 5020)

        self.open_card.clicked.connect(self.awg.open_card)
        self.STIRAP.clicked.connect(self.stirap.show)
        self.LZ.clicked.connect(self.lz.show)
        self.Rabi.clicked.connect(self.rabi.show)
        self.Sequence.clicked.connect(self.sequence.show)
        self.close_card.clicked.connect(self.awg.close_card)
        self.tcpserver.newConnection.connect(self.new_connection_slot)

    def new_connection_slot(self):
        print("new connection!")
        self.tcpsocket = self.tcpserver.nextPendingConnection()
        self.tcpsocket.readyRead.connect(self.ready_read_slot)

    def ready_read_slot(self):
        cmd = self.tcpsocket.readAll()
        print(cmd)
        # print(len(cmd.split("\n")))
        # print(cmd.split("\n")[1].split("="))
        # print()
        for i in range(len(cmd.split("\n")) - 1):
            cmd_name = cmd.split("\n")[i + 1].split("=")[0]
            value = cmd.split("\n")[i + 1].split("=")[1]

            cmd_name = str(cmd_name).split("'")[1]
            value = str(value).split("'")[1]

            if "RAWG" in cmd_name:

                if "STIRAP" in cmd_name:
                    self.stirap.show()
                    self.lz.close()
                    self.rabi.close()
                    self.sequence.close()

                    para_name = cmd_name.split("Mode_")[1].lower()
                    print("para_name: ", para_name)
                    print("value: ", value)

                    tem_widget = self.stirap.findChildren(QLineEdit, para_name)
                    tem_widget[0].setText(value)

                    if i == len(cmd.split("\n")) - 2:
                        self.stirap.done()

                elif "LZ" in cmd_name:
                    self.stirap.close()
                    self.lz.show()
                    self.rabi.close()
                    self.sequence.close()
                    para_name = cmd_name.split("Mode_")[1].lower()
                    print("para_name: ", para_name)
                    print("value: ", value)

                    tem_widget = self.lz.findChildren(QLineEdit, para_name)
                    tem_widget[0].setText(value)

                    if i == len(cmd.split("\n")) - 2:
                        self.lz.done()

                elif "RABI" in cmd_name:
                    self.stirap.close()
                    self.lz.close()
                    self.rabi.show()
                    self.sequence.close()

                    para_name = cmd_name.split("Mode_")[1].lower()
                    print("para_name: ", para_name)
                    print("value: ", value)

                    tem_widget = self.rabi.findChildren(QLineEdit, para_name)
                    tem_widget[0].setText(value)

                    if i == len(cmd.split("\n")) - 2:
                        self.rabi.done()

                elif "Sequence" in cmd_name:
                    self.stirap.close()
                    self.lz.close()
                    self.rabi.close()
                    self.sequence.show()

                    para_name = cmd_name.split("Mode_")[1].lower()
                    print("para_name: ", para_name)
                    print("value: ", value)

                    tem_widget = self.sequence.findChildren(QLineEdit, para_name)
                    tem_widget[0].setText(value)

                    if i == len(cmd.split("\n")) - 2:
                        self.sequence.done()


if __name__ == '__main__':
    # my_awg = RydbergPulseAwg()

    # start APP
    app = QApplication(sys.argv)

    # create main window
    m = Main()
    m.show()

    sys.exit(app.exec_())
