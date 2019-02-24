import face_recognition


class FaceDetector():
    def infer(self, image):
        # instance variables
        self.names = []
        self.encodings = []
        self.image = {}
        # image must be face recognition image.
        #raise NotImplementedError
        #test
        #1. get and store the face location
        #2. for every face location
        #3. a dictionnary called faces_recognized
        #pass
    def __init__(self, face_matches):
        #assume maping from a name to a picture
        #iterate through the dictionary, set up a data structure in this class to
        #make the know
        #pass
        assert len(face_matches) != 0
        for key in face_matches:
            self.names.append(key)
            try:
                encoded_face = face_recognition.face_encodings(face_matches[key])
                self.encodings.append(encoded_face)
                self.image[key] = encoded_face
            except IndexError:
                print("unable to find face")
                del self.names[-1]
