import asyncio
import websockets
import base64
from PIL import Image, ImageDraw
from io import BytesIO
from binascii import b2a_base64
from face_detector import FaceDetector
import json

lock = asyncio.Lock()
connection_num = 0
fr =  FaceDetector('./images/')

async def img_receiver(websocket, path):
    current_image_count = 0
    current_connection = None
    try:
        await lock.acquire()
        global connection_num
        current_connection = connection_num
        connection_num += 1
    finally:
        lock.release()

    try:
        while True:
            # read encoded image string
            image_str = await websocket.recv()
            # if empty, skip and wait again
            if not image_str or image_str == "null":
                continue

            # slice base64 string into data and metadata
            data_start_idx = image_str.index(",")
            image_metadata = image_str[0:data_start_idx]
            image_data = image_str[data_start_idx+1:]

            print("(conn {}, img {}) reciv: {}".format(current_connection, current_image_count, image_metadata))

            # create image
            image = BytesIO(base64.b64decode(image_data))
            name_and_coords = fr.infer_person(image)

            # encode information to json and send it
            to_send = json.dumps({'name':name_and_coords[0], 'coordinates':name_and_coords[1]})
            await websocket.send(to_send)

            # increment image count
            current_image_count += 1

    except websockets.exceptions.ConnectionClosed:
        print("Connection closed by client")

start_server = websockets.serve(img_receiver, 'localhost', 8765, max_size=None)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
