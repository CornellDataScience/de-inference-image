import face_recognition as fr
import numpy as np
import os
import re

class FaceDetector():
    def __init__(self, path_to_faces):

        self.names = []
        self.encodings = []
        self.image_dict = {}

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
                encoding = fr.face_encodings(face)
                self.encodings.append(encoding)
            except IndexError:
                print("unable to find face")
                del self.names[-1]

            #add to dictionary
            self.image_dict[name] = [encoding, face_location]


    ##Returns probability that two faces are a match (RECOMMENED IF >.93, THEN MATCH)
    def prob_of_match(self, known_face_encoding, face_encoding_to_check):
        #turn into np array
        known_face_encoding_np_array = np.array([known_face_encoding])

        #call match function and store encoding matches, lower the tolerance the more strict, 0.09 seems to work well
        encoding_matches = fr.compare_faces(known_face_encoding_np_array, face_encoding_to_check, tolerance=0.09)

        #count number of matches
        match_count = 0
        for encoding_match in encoding_matches[0]:
            if encoding_match:
                match_count += 1

        #calculate proportion of matches
        encoding_match_proportion = match_count/len(encoding_matches[0])

        #93 - 95% of all encoding checks should match
        return encoding_match_proportion #> 0.95


    ##Returns the person who matches closest with the inputted encoding and the coordinates of the location of a face
    # Format: Top Left Corner: (X,Y), Bottom Right Corner(X,Y) ==> (location[0],location[2]), (location[3]:location[1])

    def infer_people(self, image_bytes):
        face_obj = fr.load_image_file(image_bytes)
        unknown_face_encodings = fr.face_encodings(face_obj)

        #initialize empty string that will hold tuples of name of the person and their corresponding face location
        names_and_faces = []

        #Go through face encodings
        for i, unknown_face_encoding in enumerate(unknown_face_encodings):

            #if there is no face encodings (no fce in image) return empty string and empty tuple
            if len(unknown_face_encoding) == 0:
                return ('', ())

            #since there are people in this image, check who these people are- set default match
            highest_match_prob = [- 1.0, 'no match']

            #go through all people, check number of encoding-match tests that pass
            for person in self.image_dict:
                match_prob = self.prob_of_match(self.image_dict[person][0], unknown_face_encoding)

                #if more than 93% of match tests pass and is the most that has passed, make it that person
                if highest_match_prob[0] < match_prob and match_prob > 0.93:
                    highest_match_prob[0] = match_prob
                    highest_match_prob[1] = person
            
            names_and_faces.append((highest_match_prob[1], fr.face_locations(face_obj)[i]))
        
        return names_and_faces


    ##Returns whether face recognition detects a face
    def has_face(self, path_to_image):
        face = fr.load_image_file(path_to_image)
        encoding = fr.face_encodings(face)
        return len(encoding) != 0


    # Returns tuple of the coordinates of location of the face given a person's name in their original picture
    # Top Left Corner: (X,Y), Bottom Right Corner(X,Y) | (location[0],location[2]), (location[3]:location[1])
    def get_face_coordinates(self, name):
        return self.image_dict[name][1][0]


