from flask import Flask, render_template, request
import os
from random import random
from my_yolov6 import yolov6
import cv2

yolov6_model = yolov6("weights/yolov6s.pt","cpu","data/coco.yaml", 640, True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static"

@app.route("/", methods=['GET', 'POST'])
def home_page():
    if request.method == "POST":
         try:
            image = request.files['file']
            if image:
                path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
                print("Save = ", path_to_save)
                image.save(path_to_save)

                frame = cv2.imread(path_to_save)

                frame, ndet, labels = yolov6_model.infer(frame, conf_thres=0.6, iou_thres=0.45)
                result = ', '.join(labels)

                if ndet!=0:
                    cv2.imwrite(path_to_save, frame)

                    return render_template("index.html", user_image = image.filename , rand = str(random()),
                                           msg="File uploaded", ndet = ndet, result = result)
                else:
                    return render_template('index.html', msg='No object detected')
            else:
                return render_template('index.html', msg='Upload image')

         except Exception as ex:
            print(ex)
            return render_template('index.html', msg='No object detected')

    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)