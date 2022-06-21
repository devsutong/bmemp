from flask import Flask, request
from flask import render_template
import os
from tensorflow.keras.models import model_from_json
import cv2
# from IPython.display import clear_output

IMAGE_FOLDER = os.path.join('static', 'photo')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

def load_model():
    model = model_from_json(open("./models/v1/model_arch.json", "r").read())
    model.load_weights('./models/v1/model_weights.h5')
    return model

def prepare(image):
    new_array = cv2.resize(image, (180, 180))
    return new_array.reshape(-1, 180, 180, 3)

def predict(img_path):
    x = 0
    y = 0
    capture = cv2.VideoCapture(img_path)
    model = load_model()
    while (True):
        sucess, img  = capture.read()
        img = cv2.rectangle(img, (x, y), (x + 100, y + 30), (256, 256, 256))
        # prediction = model.predict(prepare(img))[0][0]*100
        prediction = model.predict(prepare(img))

        max = 0
        index = 0

        for x in range(3):
            if prediction[0][x] > max:
                max = x
        if max == 0:
            print("Prediction: BCC")
        elif max == 1:
            print("Prediction: Melanoma")
        else:
            print("Prediction: Squamous Cell Carcinoma")


        # if max == 0:
        #     cv2.putText(img, "BCC",(x+50,y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        #     # cv2.imshow("frame", img)
        # elif max == 1:
        #     cv2.putText(img, "MEL",(x+50,y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        # else:
        #     cv2.putText(img, "SCC",(x+50,y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        result = max
        return result
        # else:
        #     cv2.putText(img, "No Class?",(x+50,y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        #     cv2.imshow("frame", img)

        # clear_output(wait=True)
        # print(prediction, flush=True)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        
    # cv2.destroyAllWindows()


@app.route('/', methods=['GET', 'POST'])
def index():
    filename = os.path.join(IMAGE_FOLDER, "spreading_melanoma.jpg")
    return render_template('index.html', img = filename)

@app.route("/submit", methods = ["GET", 'POST'])
def get_prediction():
    if request.method == 'POST':
        img = request.files['image']
        img_path = "static/photo/" + img.filename
        img.save(img_path)

        result = predict(img_path)

        x = 0
        y = 0
        if result == 0:
            prediction = "BCC"
            # cv2.putText(img, "BCC",(x+50,y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            # cv2.imshow("frame", img)
        elif result == 1:
            prediction = "MEL"
            # cv2.putText(img, "MEL",(x+50,y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        else:
            prediction = "SCC"
            # cv2.putText(img, "SCC",(x+50,y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)


        
        p = prediction
    return render_template("index.html", prediction = p, img_path = img_path)

def generate_frame():
    pass

@app.route("/about")
def about_page():
	return "Developed by Risamhauni Shylla, Hiamdor Mairom et al!"

@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon.ico')


if __name__ == '__main__':
   app.run()

