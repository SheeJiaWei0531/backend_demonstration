
import base64, logging, json, os, pathlib, sys, time, traceback, cv2, requests
from datetime import datetime
import cv2
import requests
from flask import Flask, request
from flask_cors import CORS


import numpy as np

project_dir = pathlib.Path(__file__).absolute().parent.absolute()
log_dirt = os.path.join(project_dir, "history.log")
sys.path.insert(0, project_dir.as_posix())
from utils.Detectors import save_base64_to_mp4



logging.basicConfig(
        filename=log_dirt,
        filemode='a',
        level=logging.INFO,  
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

logging.info("Start server")

app = Flask(__name__)
CORS(app)

MEGABYTE = (2 ** 10) ** 2
app.config['MAX_CONTENT_LENGTH'] = None
app.config['MAX_FORM_MEMORY_SIZE'] = 50 * MEGABYTE
app.config['MAX_FORM_PARTS'] = 2000


@app.route("/health", methods = ["GET"])
def health():
    logging.info("healthcheck")
    return {"code": 200}



@app.route("/analyze",  methods = ['POST'])
def analyze():

   




    video_path = None
    data = request.get_json()
    logging.info(data)   
    email_address = data.get("email", None)
    b64str_video = data.get("videob64", None)
    user_name = data.get("name", None)
    if b64str_video:
        video_path = save_base64_to_mp4(b64str= b64str_video)
            
    
    
    
    
    return {"code": 200}