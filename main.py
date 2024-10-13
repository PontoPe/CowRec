# YOLO object detection
import cv2 as cv
import numpy as np
import time
import time
import random

cowDetected = 0

# Simulação de Pin e PWM para controlar o servo
class Pin:
    def __init__(self, num, mode=None):
        self.num = num
        self.mode = mode

class PWM:
    def __init__(self, pin, freq=50):
        self.pin = pin
        self.freq = freq

    def duty(self, value):
        print(f"Simulando movimento do servo no pino {self.pin.num} com duty: {value}")

# Simulação de rede Wi-Fi
class network:
    class WLAN:
        def __init__(self, mode):
            pass

        def active(self, status):
            pass

        def connect(self, ssid, password):
            print(f"Simulando conexão Wi-Fi à rede: {ssid}")

        def isconnected(self):
            return True

        def ifconfig(self):
            return ("192.168.1.100", "255.255.255.0", "192.168.1.1", "8.8.8.8")

# Simulação do cliente MQTT
class MQTTClient:
    def __init__(self, client_id, broker):
        self.client_id = client_id
        self.broker = broker

    def connect(self):
        print(f"Simulando conexão ao broker MQTT {self.broker}")

    def publish(self, topic, msg):
        print(f"Simulando publicação no tópico {topic}: {msg}")

    def subscribe(self, topic):
        print(f"Simulando inscrição no tópico {topic}")

    def check_msg(self):
        print("Simulando verificação de mensagens MQTT")

# Simulação do sensor de peso HX711
class HX711:
    def __init__(self, dout_pin, sck_pin):
        self.dout_pin = dout_pin
        self.sck_pin = sck_pin

    def tare(self):
        print("Simulando taragem da balança")

    def get_value(self):
        return random.randint(0, 15000)  # Simulação de peso aleatório

# Configurações simuladas
SSID = 'SEU_SSID'
PASSWORD = 'SUA_SENHA'
MQTT_BROKER = 'broker.hivemq.com'
CLIENT_ID = "esp32_simulado"
TOPIC_PESO = b'esp32/peso'
TOPIC_FOTO = b'esp32/foto'
TOPIC_SERVO = b'esp32/servo'

# Função para conectar ao Wi-Fi (simulada)
def connect_wifi():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.connect(SSID, PASSWORD)
    while not wlan.isconnected():
        time.sleep(1)
    print('WiFi conectado! IP:', wlan.ifconfig())

# Função para controlar o servo
servo = PWM(Pin(15))

def move_servo(angle):
    duty = int((angle / 180 * 1023) + 40)
    servo.duty(duty)

# Função para simular tirar uma foto
def take_photo():
    print("Foto capturada!")
    return "imagem_base64_simulada"

# Função principal
def main():
    client = MQTTClient(CLIENT_ID, MQTT_BROKER)
    client.connect()
    client.subscribe(TOPIC_SERVO)

    hx = HX711(Pin(12), Pin(14))  # Correção na inicialização da classe HX711
    hx.tare()

    while True:
        peso = hx.get_value()
        print(f"Peso detectado: {peso}")
        client.publish(TOPIC_PESO, str(peso))

        if peso > 10000:  # Limite de peso
            foto = take_photo()
            yolo(foto)
            client.publish(TOPIC_FOTO, foto)
            if cowDetected == 1:
                move_servo(90)
                client.publish(TOPIC_SERVO, "90")

        client.check_msg()  # Verificar por mensagens MQTT
        time.sleep(1)

main()


def yolo(img):
    global cowDetected
    cowDetected = 0
    cv.imshow('window',  img)
    cv.waitKey(1)

    # Load names of classes and get random colors
    classes = open('YOLO_packs/coco.names').read().strip().split('\n')
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

    # Give the configuration and weight files for the model and load the network.
    net = cv.dnn.readNetFromDarknet('YOLO_packs/yolov3.cfg', 'YOLO_packs/yolov3.weights')
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    # net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    # determine the output layer
    layers = net.getLayerNames()
    ln = [layers[i - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the image
    blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    r = blob[0, 0, :, :]

    cv.imshow('blob', r)
    text = f'Blob shape={blob.shape}'
    #cv.displayOverlay('blob', text)
    cv.waitKey(1)

    net.setInput(blob)
    t0 = time.time()
    outputs = net.forward(ln)
    t = time.time()
    print('time=', t-t0)

    print(len(outputs))
    for out in outputs:
        print(out.shape)

    def trackbar2(x):
        confidence = x/100
        r = r0.copy()
        for output in np.vstack(outputs):
            if output[4] > confidence:
                x, y, w, h = output[:4]
                p0 = int((x-w/2)*416), int((y-h/2)*416)
                p1 = int((x+w/2)*416), int((y+h/2)*416)
                cv.rectangle(r, p0, p1, 1, 1)
        cv.imshow('blob', r)
        text = f'Bbox confidence={confidence}'
        #cv.displayOverlay('blob', text)

    r0 = blob[0, 0, :, :]
    r = r0.copy()
    cv.imshow('blob', r)
    cv.createTrackbar('confidence', 'blob', 50, 101, trackbar2)
    #trackbar2(50)

    boxes = []
    confidences = []
    classIDs = []
    h, w = img.shape[:2]

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.6:
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                box = [x, y, int(width), int(height)]
                boxes.append(box)
                confidences.append(float(confidence))
                classIDs.append(classID)

    indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in colors[classIDs[i]]]
            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
            cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            print(f"Detected object: {classes[classIDs[i]]}, Confidence: {confidences[i]:.3f}")
            if classes[classIDs[i]] == 'cow':
                cowDetected = 1
        if cowDetected == 1:
            print('Cow detected!!!!!!!')
