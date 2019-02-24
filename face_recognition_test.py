
# coding: utf-8

# In[19]:


import face_recognition as fr
from PIL import Image
import os
import re
import numpy as np


# # Temporary setup
# 
# ## Setup Known People

# In[20]:


#Load all files in directory 
directory = os.listdir("./Desktop/images")
file_path = "./Desktop/images/"

#regex to parse out name from filename
file_name = re.compile("(^.+)\....$") 

names = []
encodings=[]
image_dict = {}


# In[21]:


for image_name in directory: 
    #skip over .DS_Store
    if(image_name == '.DS_Store'):
        continue
        
    #add names to names list
    match = file_name.search(image_name)
    name = match.group(1)
    names.append(name)

    #detect faces
    face = fr.load_image_file(file_path + image_name)
    encoding = fr.face_encodings(face)
    encodings.append(encoding)
    
    #store encodings
    image_dict[name] = encoding


# ## Setup unknown people

# In[22]:


#Load all files in directory 
directory_u = os.listdir("./Desktop/unknown/")
file_path_u = "./Desktop/unknown/"

unknown_names = []
unknown_encodings=[]
unknown_image_dict = {}


# In[23]:


for image_name in directory_u: 
    #skip over .DS_Store
    if(image_name == '.DS_Store'):
        continue
        
    #add names to names list
    match = file_name.search(image_name)
    name = match.group(1)
    unknown_names.append(name)

    #detect faces
    face = fr.load_image_file(file_path_u + image_name)
    encoding = fr.face_encodings(face)
    unknown_encodings.append(encoding)
    
    #store encodings
    unknown_image_dict[name] = encoding


# # Methods
# ### Probability of match between two encodings

# In[24]:


##Returns probability that two faces are a match (RECOMMENED IF >.95, THEN MATCH)
def prob_of_match(known_face_encoding, face_encoding_to_check): 
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


# ### Infer unknown person 

# In[27]:


##Returns the person who matches closest with the inputted encoding
def infer_person(unknown_face_encoding):
    #default
    highest_match_prob = [- 1.0, 'no match']
    
    for person in image_dict:
        match_prob = prob_of_match(image_dict[person], unknown_face_encoding)

        if highest_match_prob[0] < match_prob and match_prob > 0.93:
            highest_match_prob[0] = match_prob
            highest_match_prob[1] = person

    return highest_match_prob[1]


# # Fun time

# In[29]:


print(infer_person(unknown_image_dict['3']))

