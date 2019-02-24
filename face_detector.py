import face_recognition as fr
import numpy as np
import os
import re



class FaceDetector():
    def __init__(self, path_to_faces):

        self.names = []
        self.encodings = []
        self.image_dict = {}

        directory = os.listdir(path_to_faces)
        assert len(directory) != 0
        file_path = path_to_faces

        file_name = re.compile("(^.+)\....$")
        for image_name in directory:

            if(image_name == '.DS_Store'):
                continue

            #add names to names list
            match = file_name.search(image_name)
            name = match.group(1)
            self.names.append(name)

            #detect faces
            try:
                face = fr.load_image_file(file_path + image_name)
                encoding = fr.face_encodings(face)
                self.encodings.append(encoding)
            except IndexError:
                print("unable to find face")
                del self.names[-1]

            #add to dictionary
            self.image_dict[name] = encoding

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



    ##Returns the person who matches closest with the inputted encoding
    def infer_person(self, unknown_face_encoding):
        #default
        highest_match_prob = [- 1.0, 'no match']

        for person in self.image_dict:
            match_prob = prob_of_match(self.image_dict[person], unknown_face_encoding)

            if highest_match_prob[0] < match_prob and match_prob > 0.93:
                highest_match_prob[0] = match_prob
                highest_match_prob[1] = person

        return highest_match_prob[1]


    ##Returns whether face recognition detects a face
    def has_face(self, path_to_image):
        face = fr.load_image_file(path_to_image)
        encoding = fr.face_encodings(face)
        return len(encoding) != 0
