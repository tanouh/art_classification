
import os
from datetime import datetime

def log_message(message, log_path):
    timestamp = datetime.now().strftime("%d/%m/%y, %H:%M:%S")
    with open(log_path, "a") as f:
        f.write(f"{timestamp} : {message} \n")