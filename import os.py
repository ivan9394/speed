import os
import sys
import speed
import datetime
import threading
import uuid



UPLOAD_FOLDER = os.getcwd() + "/upload/video"
ALLOWED_EXTENSIONS = {'mp4','mov','avi', 'wmv', 'flv', 'mkv', 'm4v'}
# SQLite URI compatible
prefix = 'sqlite:///'


