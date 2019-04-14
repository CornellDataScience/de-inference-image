import face_recognition as fr
import numpy as np
import os
import re

class FaceDetector():
    def __init__(self, path_to_faces):

        self.names = []
        self.encodings = []
        self.known_people_dict = {}

        #path to faces this instance of FaceDetector already knows
        directory = os.listdir(path_to_faces)
        assert len(directory) != 0
        file_path = path_to_faces

        #regex to parse out names from file names
        file_name = re.compile("(^.+)\.(png|jpeg|jpg)$")
        for image_name in directory:

            #add names to names list, if not image file, skip
            match = file_name.search(image_name)
            if match == None: continue
            name = match.group(1)
            self.names.append(name)

            #add face encoding to names list, if cannot find a face throws IndexError
            #*PRECONDITION: ALL IMAGES MUST HAVE 1 FACE
            try:
                face = fr.load_image_file(file_path + image_name)
                face_location = fr.face_locations(face)
                encoding = fr.face_encodings(face)[0]
                self.encodings.append(encoding)
            except IndexError:
                print("unable to find face")
                del self.names[-1]

            #add to dictionary
            self.known_people_dict[name] = [encoding, face_location]


    ##Returns the person who matches closest with the inputted encoding and the coordinates of the location of a face
    # Format: Top Left Corner: (X,Y), Bottom Right Corner(X,Y) ==> (location[0],location[2]), (location[3]:location[1])
    def infer_people(self, unknown_image_bytes):
        
        #names_and_location = [([list of possible people for face1], face1 location), ...]
        names_and_location = []

        #process image
        face_obj = fr.load_image_file(unknown_image_bytes)
        unknown_face_encodings = fr.face_encodings(face_obj)
        
        #if there is no one in encodings, return no one in image
        if len(unknown_face_encodings) == 0:
            return None 

        all_face_locations = fr.face_locations(face_obj)

        #go through all unknown face encodings in current image, compare faces, 
        #TODO: Research if you can thread for each person here?
        for i, unknown_face_encoding in enumerate(unknown_face_encodings):
            test_results = fr.compare_faces(self.encodings, unknown_face_encoding, tolerance=0.6)
            
            #TODO: possibly speed up with numpy array? possible_names = np.empty(1, dtype='string_')
            possible_names = []
            
            #TODO: make sure this is right (if names is in the same order as encodings was ran)
            for j, test_result in enumerate(test_results):
                if test_result:
                    possible_names.append(self.names[j])
                    
            #if no match, person is unknown
            if len(possible_names) == 0:
                possible_names.append('Unknown')
                
            names_and_location.append((possible_names, all_face_locations[i]))

        return names_and_location
        

    ##Returns whether face recognition detects a face
    def has_face(self, path_to_image):
        face = fr.load_image_file(path_to_image)
        encoding = fr.face_encodings(face)
        return len(encoding) != 0
