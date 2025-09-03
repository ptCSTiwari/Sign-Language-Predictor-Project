from flask import Flask, render_template, Response, request, flash
from keras.models import model_from_json
import cv2
import numpy as np

# Load model architecture
json_file = open(r"C:\Users\pc\Music\last\signlanguagedetectionmodel128x128.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

# Load model weights
model.load_weights(r"C:\Users\pc\Music\last\signlanguagedetectionmodel128x128.h5")
print("✅ Model loaded successfully!")

# Labels (24 total classes)
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
          'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
          'U', 'W', 'X', 'Y', 'Z', 'blank']

# Feature extraction function
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 128, 128, 1)
    return feature / 255.0

# Flask app setup
app = Flask(__name__)
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Draw ROI rectangle
        cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 1)

        # Crop & preprocess
        cropframe = frame[40:300, 0:300]
        cropframe = cv2.cvtColor(cropframe, cv2.COLOR_BGR2GRAY)
        cropframe = cv2.resize(cropframe, (128, 128))
        cropframe = extract_features(cropframe)

        # Prediction
        pred = model.predict(cropframe, verbose=0)
        idx = np.argmax(pred)

        # Safe prediction
        if idx < len(labels):
            prediction = labels[idx]
        else:
            prediction = "Unknown"

        # Display prediction
        cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)
        accu = "{:.2f}".format(np.max(pred) * 100)
        cv2.putText(frame, f'{prediction} {accu}%', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


        # Encode for Flask stream
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start')
def start():
    return render_template('start.html')

@app.route("/about")
def about():
    return render_template("about.html")

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')

        # Save/process message (log, database, email, etc.)
        print(f"Name: {name}, Email: {email}, Message: {message}")

        flash("Thank you! Your message has been sent successfully.", "success")
        return redirect(url_for('contact'))
    
    return render_template('contact.html')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
