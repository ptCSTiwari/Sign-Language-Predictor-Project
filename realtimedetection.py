from keras.models import model_from_json
import cv2
import numpy as np

# Load the model architecture and weights
json_file = open(r"C:\Users\pc\Music\last\signlanguagedetectionmodel128x128.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights(r"C:\Users\pc\Music\last\signlanguagedetectionmodel128x128.h5")

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 128, 128, 1)
    return feature / 255.0

cap = cv2.VideoCapture(0)
label =['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'U', 'W', 'X', 'Y', 'Z', 'blank']

while True:
    _, frame = cap.read()
    cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 1)
    cropframe = frame[40:300, 0:300]
    cropframe = cv2.cvtColor(cropframe, cv2.COLOR_BGR2GRAY)
    cropframe = cv2.resize(cropframe, (128, 128))
    cropframe = extract_features(cropframe)
    
    pred = model.predict(cropframe)
    prediction_label = label[pred.argmax()]
    
    cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)
    
    if prediction_label == 'blank':
        cv2.putText(frame, " ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        accu = "{:.2f}".format(np.max(pred) * 100)
        cv2.putText(frame, f'{prediction_label}  {accu}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow("output", frame)
    
    key = cv2.waitKey(1)  # Check for key press
    if key != -1:  # If any key is pressed, exit the loop
        break

cap.release()
cv2.destroyAllWindows()