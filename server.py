from http.server import HTTPServer, BaseHTTPRequestHandler
from io import BytesIO
from face_detector import FaceDetector
from binascii import b2a_base64
import base64
import json

fr =  FaceDetector('./images/')

class ImageProcessing(BaseHTTPRequestHandler):

    def do_POST(self):
        #setup
        content_length = int(self.headers['Content-Length'])
        json_body = self.rfile.read(content_length)

        #retrieve base64 image string from json dictionary (only thing in dictionary)
        json_dict = json.loads(json_body)
        body = json_dict.get('image')

        self.send_response(200)
        self.end_headers()

        # slice base64 string into data and metadata
        data_start_idx = body.index(",")
        # image_metadata = body[0:data_start_idx]
        image_data = body[data_start_idx+1:]

        # create image
        image_bytes = BytesIO(base64.b64decode(image_data))

        # extract faces
        names_and_coords = fr.infer_people(image_bytes)
        
        #create a dictionary out of everyone with keys=name and values=coordinates of face
        face_data = []
        for name_and_coord in names_and_coords:
            face_data.append({"name": name_and_coord[0], "coordinates": name_and_coord[1]})

        face_data_bytes = bytearray(json.dumps(face_data), encoding="utf-8")
        self.wfile.write(face_data_bytes)


#This creates the HTTP server
#TODO: Change from local host to something else
httpd = HTTPServer(('localhost', 8000), ImageProcessing)
httpd.serve_forever()
