#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import copy
import pandas as pd
import time
import math

def isNaN(a):
    return a != a


# In[16]:


delta_h_limit = 3
delta_w_limit = 3

def find_nearest_height_idx(user_info, user_idx, height, start, last, idx, delta_h_tolerance):
    mid = int(((start + last) / 2))

    if start > last:
        return None
    if (abs(user_info.iloc[idx[mid]]['신장'] - user_info.iloc[user_idx]['신장'])) <= delta_h_tolerance:
        return mid
    if user_info.iloc[idx[mid]].loc['신장'] > height:
        return find_nearest_height_idx(user_info, user_idx, height, start, mid - 1, idx, delta_h_tolerance)
    elif user_info.iloc[idx[mid]].loc['신장'] < height:
        return find_nearest_height_idx(user_info, user_idx, height, mid + 1, last, idx, delta_h_tolerance)
    else:
        return None

def find_nearest_weight_idx(user_info, user_idx, weight, start, last, delta_h_tolerance, delta_w_tolerance):
    mid = int(((start + last) / 2))
  
    if start > last:
        return None
    if get_distance(user_info, user_idx, mid, delta_h_tolerance, delta_w_tolerance) != None:
        return mid
    if user_info.iloc[mid].loc['체중'] > weight:
        return find_nearest_weight_idx(user_info, user_idx, weight, start, mid - 1, delta_h_tolerance, delta_w_tolerance)
    elif user_info.iloc[mid].loc['체중'] < weight:
        return find_nearest_weight_idx(user_info, user_idx, weight, mid + 1, last, delta_h_tolerance, delta_w_tolerance)
    else:
        return None

    
def get_D(user_info, idx, delta_h_tolerance = delta_h_limit, detla_w_tolerance = delta_w_limit):
    START = time.time()
    distance_matrix = pd.DataFrame(index = user_info.index, columns = user_info.index)
  
    breaker = False
   
    for user_idx in range(len(user_info.index)):
        print(user_idx)
        print(time.strftime('%c', time.localtime(time.time())))
        for i in range(len(idx)):
            start = idx[i]
            if i == (len(idx) - 1):
                last = user_info.index.size - 1
            else:
                last = idx[i+1] - 1
            if user_idx > last:
                continue
            if user_idx > start:
                start = user_idx
                
            if user_info.iloc[start]['신장'] >= user_info.iloc[user_idx]['신장'] + delta_h_tolerance:
                break
        
            mid = find_nearest_weight_idx(user_info, user_idx, user_info.iloc[user_idx].loc['체중'], start, last) 

            if mid == None:
                continue
            else:
                left_side = mid

                while start <= left_side:

                    D = get_distance(user_info, user_idx, left_side) 
                    if D != None:
                        distance_matrix.iloc[user_idx, left_side], distance_matrix.iloc[left_side, user_idx] = D
                        left_side -= 1
                    else:
                        break
                
                right_side = mid
         
                while right_side <= last:
                    D = get_distance(user_info, user_idx, right_side) 
                    if D != None:
                        distance_matrix.iloc[user_idx, right_side], distance_matrix.iloc[right_side, user_idx] = D
                        right_side += 1
                    else:
                        break
    print("total", time.time()-START)
    return distance_matrix

def get_height_idx(user_info, user_id, idx, delta_h_tolerance = delta_h_limit, detla_w_tolerance = delta_w_limit):
    user_idx = list(user_info.index).index(user_id)
    
    mid = find_nearest_height_idx(user_info, user_idx, user_info.loc[user_id]['신장'], 0, len(idx) - 1, idx, delta_h_tolerance)
    
    if mid == None:
        return None
  
    left_side = mid
    while 0 <= left_side:
        if (abs(user_info.iloc[idx[left_side]]['신장'] - user_info.iloc[user_idx]['신장'])) <= delta_h_tolerance:
            left_side -= 1
        else:
            break

    right_side = mid
    while right_side < len(idx):
        if (abs(user_info.iloc[idx[right_side]]['신장'] - user_info.iloc[user_idx]['신장'])) <= delta_h_tolerance:
            right_side += 1
        else:
            break

    idx = idx[left_side + 1 : right_side]
    
    return copy.deepcopy(idx)

def get_personal_D(user_info, user_id, height_idx, delta_h_tolerance = delta_h_limit, detla_w_tolerance = delta_w_limit):
    distance_matrix = pd.DataFrame(index = [user_id], columns = user_info.index)
    
    user_idx = list(user_info.index).index(user_id)
    start_idx = []
    
    for i in range(len(height_idx)):
        start = height_idx[i]
        if i == (len(height_idx) - 1):
            last = user_info.index.size - 1
        else:
            last = height_idx[i+1] - 1
 
        mid = find_nearest_weight_idx(user_info, user_idx, user_info.iloc[user_idx].loc['체중'], start, last, delta_h_tolerance, detla_w_tolerance) 

        if mid == None:
            continue
        else:
            left_side = mid

            while start <= left_side:

                D = get_distance(user_info, user_idx, left_side, delta_h_tolerance, detla_w_tolerance) 
                if D != None:
                    distance_matrix.loc[user_id].iloc[left_side], rmsidsjgdmsrj = D
                    left_side -= 1
                else:
                    break
                    
            start_idx.append(left_side + 1)
            
            right_side = mid

            while right_side <= last:
                D = get_distance(user_info, user_idx, right_side, delta_h_tolerance, detla_w_tolerance) 
                if D != None:
                    distance_matrix.loc[user_id].iloc[right_side], rmsidsjgdmsrj = D
                    right_side += 1
                else:
                    break
    
    return distance_matrix, start_idx
         
def get_distance(user_info, user1_idx, user2_idx, delta_h_tolerance, delta_w_tolerance):
    user1_height = user_info.iloc[user1_idx].loc['신장']
    user1_weight = user_info.iloc[user1_idx].loc['체중']
    user2_height = user_info.iloc[user2_idx].loc['신장']
    user2_weight = user_info.iloc[user2_idx].loc['체중']
    
    delta_height = user1_height - user2_height
    delta_weight = user1_weight - user2_weight
    
    if delta_w_tolerance == 0:
        if delta_weight != 0:
            return None
        else:
            distance = abs(delta_height)
    else:
        distance = math.sqrt((delta_height ** 2) + ((delta_h_tolerance / delta_w_tolerance * delta_weight) ** 2))
    
    if distance > delta_h_tolerance:
        return None
    
    return distance, distance

def get_user_profile(user_id, distance_matrix, starT_idx, rating_info, clothing_info, size_category):
    profiles = pd.DataFrame(index = ['0'], columns = size_category)
    for s_category in size_category:
            profiles.iloc[0][s_category] = pd.DataFrame(index = ['아이고'], columns = ['의미없다'])
        
    for start_idx in starT_idx:
        user_idx = start_idx
      
        while True:
            distance = distance_matrix.loc[int(user_id)].iloc[user_idx]
            if isNaN(distance) == True:
                break
            for clothing in rating_info.columns: # clothing : object
                if isNaN(rating_info.iloc[user_idx].loc[clothing]) == True:
                    continue
                for s_category in size_category:
                    profile = profiles.iloc[0][s_category]
                    shopper = str(rating_info.index[user_idx])
       
                    size = str(clothing_info.loc[int(clothing)][s_category])
                  
                    if (shopper in list(profile.columns)) == False:
                        profile.loc[:, shopper] = 0
                    
                    if (rating_info.iloc[user_idx].loc[clothing] == 1) or (rating_info.iloc[user_idx].loc[clothing] == 3):
                        if (size in list(profile.index)) == False:
                            profile.loc[size] = 0
                            profile.loc[size, shopper] = -1
                        elif profile.loc[size, shopper] == None: 
                            profile.loc[size, shopper] = -1
                        else:
                            profile.loc[size, shopper] -= 1
                            
                    elif rating_info.iloc[user_idx].loc[clothing] == 2:
                        if (size in list(profile.index)) == False:
                            profile.loc[size] = 0
                            profile.loc[size, shopper] = 1
                        elif profile.loc[size, shopper] == None: 
                            profile.loc[size, shopper] = 1
                        else:
                            profile.loc[size, shopper] += 1

            user_idx += 1  
            
    for s_category in size_category:
        profiles.iloc[0][s_category].drop('아이고', inplace = True)
        profiles.iloc[0][s_category].drop('의미없다', axis = 1, inplace = True)
        profiles.iloc[0][s_category] = profiles.iloc[0][s_category].sort_index()
        profiles.iloc[0][s_category].loc[:, 'sum'] = [0, ]
        for i in profiles.iloc[0][s_category].index:
            profiles.iloc[0][s_category].loc[i, 'sum'] = sum(profiles.iloc[0][s_category].loc[i])
    
    return profiles

def get_estimate_size(user_id, distance_matrix, starT_idx, rating_info, clothing_info, size_category):
    similar_user_rating_info = pd.DataFrame(columns = ['user_id', 'distance', 'influence'] + list(clothing_info.columns[0:4]))
    total_influence = 0

    #rating_num = 0
    
    for start_idx in starT_idx:
        user_idx = start_idx
        while True:
            distance = distance_matrix.loc[int(user_id)].iloc[user_idx]
            if isNaN(distance) == True:
                break
            for clothing in rating_info.columns:
                #if isNaN(rating_info.iloc[user_idx].loc[clothing]) == False:
                #    rating_num += 1
                if rating_info.iloc[user_idx].loc[clothing] == 2: # 적당함 이면
                    size = list(clothing_info.loc[int(clothing)][size_category[0] : size_category[-1]])
                    similar_user_rating_info.loc[similar_user_rating_info.index.size] = [rating_info.index[user_idx], distance, None] + size
                    if distance == 0:
                        distance = 0.9
                    total_influence += (1 / distance)
            user_idx += 1
                
    for i in similar_user_rating_info.index:
        distance = similar_user_rating_info.loc[i]['distance']
        if distance == 0:
            distance = 0.9
        influence = 1 / distance
        similar_user_rating_info.loc[i]['influence'] = influence / total_influence

    estimate = []
    
    for s_category in size_category:
        estimate_value = 0
        for user in similar_user_rating_info.loc[:, s_category].index:
            estimate_value += (similar_user_rating_info.loc[user][s_category] * similar_user_rating_info.loc[user]['influence'])
        estimate.append(estimate_value)
        
    #print("총 순회한 rating 수 : ", rating_num,"개")
    return estimate, similar_user_rating_info

def recommed_by_distance(user_size, clothing_list, clothing_info, size_category): # clothing_list = [String, ... ] : ID
    min_error = float('inf')
    
    for clothing in clothing_list:
        clothing_size = list(clothing_info.loc[int(clothing)][size_category[0]:size_category[-1]])
        error = 0
        for size in size_category:
            error += ((user_size.loc[size] - clothing_info.loc[int(clothing)][size]) ** 2)
        error = math.sqrt(error)
        print("clothing ID : ",clothing)
        print("size : ",clothing_size)
        print("user size : ", list(user_size))
        print("error : ", error, end = "\n\n")
        if error < min_error:
            min_error = error
            best_fit = clothing
            
    return best_fit

def recommend_by_cosine(user_size, clothing_list, clothing_info, size_category): # clothing_list = [String, ... ] : ID
    cosine = -1
    
    for clothing in clothing_list:
        clothing_size = list(clothing_info.loc[int(clothing)][size_category[0]:size_category[-1]])
       
        clothing_vector_norm = math.sqrt(sum(np.multiply(clothing_size, clothing_size)))
        user_vector_norm = math.sqrt(sum(np.multiply(user_size, user_size)))
        inner_product = sum(np.multiply(user_size, clothing_size))
        
        cos = inner_product / (user_vector_norm * clothing_vector_norm)
        print("clothing ID : ",clothing)
        print("size : ",clothing_size)
        print("user size : ", list(user_size))
        print("cos : ", cos, end = "\n\n")
        if cos > cosine:
            cosine = cos
            best_fit = clothing

    return best_fit

def get_error_and_evaluation(user_id, user_size, rating_info, clothing_info, size_category): 
    Err_Evaluation = pd.DataFrame(columns = ['error', 'rating'])
    clothings = []
    for clothing in rating_info.columns:
        if isNaN(rating_info.loc[user_id][clothing]) == False:
            clothings.append(clothing)
    for clothing in clothings:
        error = 0
        clothing_size = clothing_info.loc[int(clothing)][size_category[0] : size_category[-1]]
        for s_category in size_category:
            error += ((user_size.loc[s_category] - clothing_size.loc[s_category]) ** 2)
        error = math.sqrt(error)
        Err_Evaluation.loc[clothing] = [error, rating_info.loc[int(user_id)][clothing]]
        
    return Err_Evaluation


    


# In[17]:


data = pd.read_csv('/home/csj3684/Business-Intelligence13/user_table.csv', engine='python')
user_info = pd.DataFrame(index = list(data.iloc[:, 0]), columns = list(data.columns[1:len(data.columns)]))
for col in user_info.columns:
    user_info.loc[:, col] = list(data.loc[:, col])
    
data = pd.read_csv('/home/csj3684/Business-Intelligence13/product_table.csv', engine='python')
size_category = list(data.columns[1:5])
clothing_info = pd.DataFrame(index = list(data.iloc[:, 0]), columns = list(data.columns[1:len(data.columns)]))
for col in clothing_info.columns:
    clothing_info.loc[:, col] = list(data.loc[:, col])

data = pd.read_csv('/home/csj3684/Business-Intelligence13/user_product_table.csv', engine='python')
rating_info = pd.DataFrame(index = list(data.iloc[:, 0]), columns = list(data.columns[1:len(data.columns)]))
for col in rating_info.columns:
    rating_info.loc[:, col] = list(data.loc[:, col])
    
user_size_info = pd.DataFrame(index = user_info.index, columns = size_category)

user_profile_info = pd.DataFrame(columns = size_category)

idx = []
height = 0
for user_idx in range(user_info.index.size):
    if user_info.iloc[user_idx].loc['신장'] != height:
        height = user_info.iloc[user_idx].loc['신장']
        idx.append(user_idx)


# In[18]:


user_id = "100185008"
delta_h_tolerance = 3
delta_w_tolerance = 3


# In[24]:


h_limit = 0
w_limit = 0
while True:
    h_limit += 1
    print(h_limit, w_limit)

    distance_matrix, start_idx = get_personal_D(user_info, int(user_id), height_idx, h_limit, w_limit)
    
    num = 0
    for user in distance_matrix.columns:
        if isNaN(distance_matrix.loc[int(user_id)][user]) == False:
             num += 1
    print("사용 될 수 있는 user 수 : ",num, "명")
    
    user_size_info.loc[int(user_id)], log = get_estimate_size(int(user_id), distance_matrix, start_idx, rating_info, clothing_info, size_category)
    print("추정에 사용된 rating 수 : ", log.index.size, "개")
    
    if log.index.size >= 100:
        break
        
    w_limit += 1
    print(h_limit, w_limit)
    
    distance_matrix, start_idx = get_personal_D(user_info, int(user_id), height_idx, h_limit, w_limit)

    num = 0
    for user in distance_matrix.columns:
        if isNaN(distance_matrix.loc[int(user_id)][user]) == False:
             num += 1
    print("사용 될 수 있는 user 수 : ",num, "명")
    
    user_size_info.loc[int(user_id)], log = get_estimate_size(int(user_id), distance_matrix, start_idx, rating_info, clothing_info, size_category)
    print("추정에 사용된 rating 수 : ", log.index.size, "개")
    
    if log.index.size >= 100:
        break
        


# In[19]:


height_idx = get_height_idx(user_info, int(user_id), idx, delta_h_tolerance, delta_w_tolerance)


# In[20]:


print(height_idx)


# In[21]:


START = time.time()
distance_matrix, start_idx = get_personal_D(user_info, int(user_id), height_idx, delta_h_tolerance, delta_w_tolerance)
print("걸린시간 : ", round(time.time() - START, 2), "초")


# In[22]:


print(start_idx)                         


# In[23]:


#b = 0
for i in distance_matrix.index:
    a = 0
    for j in distance_matrix.columns:
        if isNaN(distance_matrix.loc[i, j]) == False:
            a += 1
    print("사용 될 수 있는 user 수 : ",a, "명")
#    b += a
#print("b",b)


# In[24]:


user_profile_info.loc[user_id] = get_user_profile(int(user_id), distance_matrix, start_idx, rating_info, clothing_info, size_category).iloc[0]


# In[28]:


user_profile_info.loc[user_id]['기장'].loc[:, 'sum']
user_profile_info.loc[user_id]['어깨'].loc[:, 'sum']
#user_profile_info.loc[user_id]['가슴'].loc[:, 'sum']
#user_profile_info.loc[user_id]['소매'].loc[:, 'sum']


# In[29]:


START = time.time()
user_size_info.loc[int(user_id)], log = get_estimate_size(int(user_id), distance_matrix, start_idx, rating_info, clothing_info, size_category)
print("추정에 사용된 rating 수 : ", log.index.size, "개")
print("걸린시간 : ", round(time.time() - START, 2), "초")


# In[17]:


log.sort_values(by = 'distance')


# In[18]:


print(user_info.loc[int(user_id)])
#user_size_info.loc[int(user_id)]


# In[19]:


for user in log.loc[:,'user_id']:
    print(user_info.loc[user])


# In[20]:


recommend = recommed_by_distance(user_size_info.loc[int(user_id)], ['100024', '100044', '100066'], clothing_info, size_category)


# In[14]:


recommend = recommend_by_cosine(user_size_info.loc[int(user_id)], ['100024', '100044', '100066'], clothing_info, size_category)


# In[15]:


error_value = get_error_and_evaluation(int(user_id), user_size_info.loc[int(user_id)], rating_info, clothing_info, size_category)
error_value

