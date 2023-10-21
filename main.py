import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


# variables
num_classes = 7  # angry, disgust, fear, happy, sad, surprise, neutral
batch_size = 64
epochs = 5


# read kaggle facial expression recognition challenge dataset (fer2013.csv)
# https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge

with open("D:/facial-dataset/fer2013/fer2013.csv") as f:
    content = f.readlines()

lines = np.array(content)

num_of_instances = lines.size
print("number of instances: ", num_of_instances)
print("instance length: ", len(lines[1].split(",")[1].split(" ")))


# initialize trainset and test set
x_train, y_train, x_test, y_test = [], [], [], []


# transfer train and test set data
for i in range(1, num_of_instances):
    try:
        emotion, img, usage = lines[i].split(",")
        val = img.split(" ")
        pixels = np.array(val, 'float32')
        emotion = keras.utils.to_categorical(emotion, num_classes)

        if 'Training' in usage:
            y_train.append(emotion)
            x_train.append(pixels)
        elif 'PublicTest' in usage:
            y_test.append(emotion)
            x_test.append(pixels)
    except:
        print("", end="")


# data transformation for train and test sets
x_train = np.array(x_train, 'float32')
y_train = np.array(y_train, 'float32')
x_test = np.array(x_test, 'float32')
y_test = np.array(y_test, 'float32')

x_train /= 255  # normalize inputs between [0, 1]
x_test /= 255

x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
x_train = x_train.astype('float32')
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
x_test = x_test.astype('float32')

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# construct CNN structure
model = Sequential()
model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# 1st convolution layer
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# 2nd convolution layer
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))

# 3rd convolution layer
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))

#4th convolution layer
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))

model.add(Flatten())

# fully connected neural networks
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.add(Dense(num_classes, activation='softmax'))

# batch process
gen = ImageDataGenerator()
train_generator = gen.flow(x_train, y_train, batch_size=batch_size)

model.compile(loss='categorical_crossentropy'
              , optimizer=keras.optimizers.Adam()
              , metrics=['accuracy']
              )

fit = True

if fit == True:
    # model.fit_generator(x_train, y_train, epochs=epochs) #train for all trainset
    model.fit(train_generator, steps_per_epoch=batch_size, epochs=epochs)  # train for randomly selected one
else:
    model.load_weights('/data/facial_expression_model_weights.h5')  # load weights


# function for drawing bar chart for emotion preditions
def emotion_analysis(emotions):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))

    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')

    plt.show()


monitor_testset_results = False

if monitor_testset_results == True:
    # make predictions for test set
    predictions = model.predict(x_test)

    index = 0
    for i in predictions:
        if index < 30 and index >= 20:
            # print(i) #predicted scores
            # print(y_test[index]) #actual scores

            testing_img = np.array(x_test[index], 'float32')
            testing_img = testing_img.reshape([48, 48]);

            plt.gray()
            plt.imshow(testing_img)
            plt.show()

            print(i)

            emotion_analysis(i)
            print("----------------------------------------------")
        index = index + 1


# Function to detect and analyze facial expressions in real-time using webcam
def detect_emotions_with_webcam(model):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = roi_gray.astype('float32') / 255.0  # Convert to float and normalize
            img_pixels = np.expand_dims(img_pixels, axis=-1)  # Add the channel dimension

            custom = model.predict(np.expand_dims(img_pixels, axis=0))
            dominant_emotion = get_emotion(custom[0])

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, "Emotion: " + dominant_emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                        cv2.LINE_AA)

           # emotion_analysis(custom[0]) # result graph


        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Function to get the dominant emotion
def get_emotion(emotions):
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    return emotion_labels[np.argmax(emotions)]


# Call the function to detect emotions using webcam
detect_emotions_with_webcam(model)