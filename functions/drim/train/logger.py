# encoding: utf-8
__author__ = 'Jonas Teuwen'

import logging
import sys

def setup(
    use_stdout=True, filename=None, log_level=logging.INFO,
    redirect_stdout=False, redirect_stderr=False):
    """Configures basic logging"""
    
    log = logging.getLogger('')
    log.setLevel(log_level) 
    

    fmt = logging.Formatter(
        "%(levelname)-5s - %(asctime)s [%(name)-15s] %(message)s",
        datefmt="%y-%m-%d %H:%M:%S")
    
    # If use_stdout is True, add a stream handler for logging to stdout
    if use_stdout:
        ch = logging.StreamHandler(sys.stdout) 
        ch.setLevel(log_level) 
        ch.setFormatter(fmt)  
        log.addHandler(ch) 
    
    # If a filename is provided, add a file handler to log to that file
    if filename is not None:
        fh = logging.FileHandler(filename)  
        fh.setLevel(log_level)  
        fh.setFormatter(fmt)  
        log.addHandler(fh)  
    
    # If redirect_stderr is True, redirect stderr to a logger for error messages
    if redirect_stderr:
        stderr_logger = logging.getLogger('STDERR') 
        sl = StreamToLogger(stderr_logger, logging.ERROR)  
        sys.stderr = sl  
    
    # If redirect_stdout is True, redirect stdout to a logger for error messages
    if redirect_stdout:
        stdout_logger = logging.getLogger('STDOUT')  
        sl = StreamToLogger(stdout_logger, logging.ERROR)  
        sys.stdout = sl  


class StreamToLogger(object):
    """A fake file-like stream object that redirects writes to a logger instance. """
    
    def __init__(self, logger, log_level=logging.INFO):
        """Initializes the StreamToLogger object with a logger instance and log level"""

        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''  
    
    def write(self, buf):
        """Write the provided buffer to the logger line by line"""

        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip()) 

    def flush(self):
        """Flush the stream buffer (does nothing here)"""

        pass
