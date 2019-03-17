from http.server import HTTPServer, BaseHTTPRequestHandler
from io import BytesIO
from face_detector import FaceDetector
from binascii import b2a_base64
import base64

fr =  FaceDetector('./images/')

class ImageProcessing(BaseHTTPRequestHandler):

    def do_POST(self):
        #setup
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        self.send_response(200)
        self.end_headers()

        # slice base64 string into data and metadata
        data_start_idx = body.index(",")
        # image_metadata = body[0:data_start_idx]
        image_data = body[data_start_idx+1:]

        # create image
        response = BytesIO(base64.b64decode(image_data))

        # extract faces
        names_and_coords = fr.infer_people(response)
        
        #create a dictionary out of everyone with keys=name and values=coordinates of face
        face_data = []
        for name_and_coord in names_and_coords:
            face_data.append({"name": name_and_coord[0], "coordinates": name_and_coord[1]})

        response.write(face_data)
        self.wfile.write(response.getvalue())


#This creates the HTTP server
#TODO: Change from local host to something else
httpd = HTTPServer(('localhost', 8000), ImageProcessing)
httpd.serve_forever()
