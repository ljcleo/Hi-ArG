from multiprocessing import Process, get_context
from multiprocessing.pool import Pool


class NoDaemonProcess(Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(get_context())):
    Process = NoDaemonProcess


class NestablePool(Pool):
    def __init__(self, *args, **kwargs):
        kwargs["context"] = NoDaemonContext()
        super(NestablePool, self).__init__(*args, **kwargs)
