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


# In[205]:


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
    user_id = int(user_id)
    profiles = pd.DataFrame(index = ['0'], columns = size_category)
    for s_category in size_category:
            profiles.iloc[0][s_category] = pd.DataFrame(index = ['tmp_index'], columns = ['tmp_col'])
        
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
                        profile.loc[:, shopper] = None
                    
                    if (rating_info.iloc[user_idx].loc[clothing] == 1) or (rating_info.iloc[user_idx].loc[clothing] == 3):
                        if (size in list(profile.index)) == False:
                            profile.loc[size] = None
                            profile.loc[size, shopper] = -1
                        elif profile.loc[size, shopper] == None: 
                            profile.loc[size, shopper] = -1
                        else:
                            profile.loc[size, shopper] -= 1
                            
                    elif rating_info.iloc[user_idx].loc[clothing] == 2:
                        if (size in list(profile.index)) == False:
                            profile.loc[size] = None
                            profile.loc[size, shopper] = 1
                        elif profile.loc[size, shopper] == None: 
                            profile.loc[size, shopper] = 1
                        else:
                            profile.loc[size, shopper] += 1

            user_idx += 1  
            
    for s_category in size_category:
        profiles.iloc[0][s_category].drop('tmp_index', inplace = True)
        profiles.iloc[0][s_category].drop('tmp_col', axis = 1, inplace = True)
        profiles.iloc[0][s_category] = profiles.iloc[0][s_category].sort_index()
        profiles.iloc[0][s_category].loc[:, 'numerator'] = 0
        profiles.iloc[0][s_category].loc[:, 'denominator'] = 0
        profiles.iloc[0][s_category].loc[:, 'sum'] = None
        profiles.iloc[0][s_category].loc[:, 'num'] = 0
        for i in profiles.iloc[0][s_category].index:
            numerator = 0
            denominator = 0
            for j in profiles.iloc[0][s_category].columns:
                if profiles.iloc[0][s_category].loc[i, j] == 1:
                    profiles.iloc[0][s_category].loc[i, 'numerator'] += 1
                    profiles.iloc[0][s_category].loc[i, 'denominator'] += 1.5
                    profiles.iloc[0][s_category].loc[i, 'num'] += 1
                elif profiles.iloc[0][s_category].loc[i, j] == -1:
                    profiles.iloc[0][s_category].loc[i, 'denominator'] += 1.5
                    profiles.iloc[0][s_category].loc[i, 'num'] += 1
                    
            numerator = profiles.iloc[0][s_category].loc[i, 'numerator']
            denominator = profiles.iloc[0][s_category].loc[i, 'denominator']
            
            if denominator != 0:
                profiles.iloc[0][s_category].loc[i, 'sum'] = numerator / denominator
            else:
                profiles.iloc[0][s_category].loc[i, 'sum'] = 'NAN'
    return profiles

def get_adjusted_user_profile(user_profile):
    
    adjusted_user_profile = pd.DataFrame(index = ['0'], columns = size_category)
    for s_category in size_category:
        adjusted_user_profile.iloc[0][s_category] = pd.DataFrame(columns = ['numerator', 'denominator', 'sum', 'num'])
        
        for size in user_profile.loc[s_category].index:
            adjusted_user_profile.iloc[0][s_category].loc[size] = user_profile.loc[s_category].loc[size]
    
        s_adjusted_user_profile = adjusted_user_profile.iloc[0][s_category]
        s_user_profile = user_profile.loc[s_category]
    
        vir_size_idx = 0
        while True:
                if vir_size_idx >= user_profile.index.size - 1:
                    break

                size = s_adjusted_user_profile.index[0]
                size_above = s_adjusted_user_profile.index[vir_size_idx + 1]

                if float(size_above) - float(size) > 0.5:
                    break

                for col in ['numerator', 'denominator', 'num']:
                    s_adjusted_user_profile.loc[size, col] += s_user_profile.loc[size_above, col] 

                vir_size_idx += 1

        for size_idx in range(1, s_user_profile.index.size):
            vir_size_idx = size_idx
            while True:
                if vir_size_idx < 0:
                    break

                size = s_adjusted_user_profile.index[size_idx]
                size_below = s_adjusted_user_profile.index[vir_size_idx - 1]
             
                if float(size) - float(size_below) > 0.5:
                    break
                
                for col in ['numerator', 'denominator', 'num']:
                    s_adjusted_user_profile.loc[size, col] += s_user_profile.loc[size_below, col] 
                
                vir_size_idx -= 1

            vir_size_idx = size_idx    
            while True:
                if vir_size_idx >= s_user_profile.index.size - 1:
                    break

                size = s_adjusted_user_profile.index[size_idx]
                size_above = s_adjusted_user_profile.index[vir_size_idx + 1]
              
                if float(size_above) - float(size) > 0.5:
                    break

                for col in ['numerator', 'denominator', 'num']:
                    s_adjusted_user_profile.loc[size, col] += s_user_profile.loc[size_above, col] 
               
                vir_size_idx += 1

        for size in s_adjusted_user_profile.index:
            if s_adjusted_user_profile.loc[size, 'num'] < 5:
                s_adjusted_user_profile.drop(size, inplace = True)
                continue
            s_adjusted_user_profile.loc[size, 'sum'] = s_adjusted_user_profile.loc[size, 'numerator'] / s_adjusted_user_profile.loc[size, 'denominator']
            
    return adjusted_user_profile.iloc[0]
        
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

def get_compatibility_by_profile(user_id, clothing, user_profile_info, clothing_info, size_category):
    compatibility = pd.DataFrame(index = size_category, columns = ['value'])
    for s_category in size_category:
        size = float(clothing_info.loc[int(clothing), s_category])
              
        if (size in user_profile_info.loc[user_id, s_category].index) == True:
            compatibility.loc[s_category, 'value'] = user_profile_info.loc[user_id, s_category].loc[size, 'sum']
    return compatibility

def recommend_by_user_profile(user_id, clothing_list, user_profile_info, clothing_info, size_category):
    aa = pd.DataFrame(index = clothing_list, columns = ['compatibility', 'score'])
    
    for clothing in clothing_list:
        aa.loc[clothing]['compatibility'] = get_compatibility_by_profile(user_id, clothing, user_profile_info, clothing_info, size_category)
        
        
    for clothing in clothing_list:
        mean_compatibility = 0
        mean_n = 0
        breaker = False
        
        for s_category in size_category:
            compatibility = aa.loc[clothing, 'compatibility'] .loc[s_category, 'value']
            if isNaN(compatibility) == True:
                aa.loc[clothing, 'score'] = "알 수 없음"
                breaker = True
                break
            else:
                mean_compatibility += compatibility
                mean_n += 1
                
        if breaker == True:
            breaker = False
            continue
            
        if mean_n != 0:
            mean_compatibility /= mean_n
        else:
            mean_compatibility = NaN
        
        MSD_minus = 0
        MSD_plus = 0
        msd_m = 0
        msd_p = 0
        for s_category in size_category:
            compatibility = aa.loc[clothing, 'compatibility'].loc[s_category, 'value']
            """if isNaN(compatibility) == True:
                aa.loc[clothing, 'score'] = "알 수 없음"
                break
            else :"""
            if compatibility < mean_compatibility:
                MSD_minus += ((compatibility - mean_compatibility) ** 2)
                msd_m += 1
            else:
                MSD_plus += ((compatibility - mean_compatibility) ** 2)
                msd_p += 1
                    
        if msd_m != 0:
            MSD_minus = math.sqrt(MSD_minus / msd_m)
        if msd_p != 0:
            MSD_plus = math.sqrt(MSD_plus / msd_p)
        if (MSD_minus + MSD_plus) == 0:
            aa.loc[clothing, 'score'] = mean_compatibility
        else:
            aa.loc[clothing, 'score'] = mean_compatibility - (MSD_minus * (MSD_minus / (MSD_minus + MSD_plus)))
    
    score = 0
    recommend = None
    for clothing in aa.index:
        if aa.loc[clothing, 'score'] != "알 수 없음":
            if aa.loc[clothing, 'score'] > score:
                recommend = clothing
                score = aa.loc[clothing, 'score']
    
    return recommend, aa
        
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


    


# In[58]:


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
adjusted_user_profile_info = pd.DataFrame(columns = size_category)

idx = []
height = 0
for user_idx in range(user_info.index.size):
    if user_info.iloc[user_idx].loc['신장'] != height:
        height = user_info.iloc[user_idx].loc['신장']
        idx.append(user_idx)


# In[4]:


user_id = "100265001"
delta_h_tolerance = 2
delta_w_tolerance = 2


# In[5]:


height_idx = get_height_idx(user_info, int(user_id), idx, delta_h_tolerance, delta_w_tolerance)


# In[6]:


print(height_idx)


# In[7]:


START = time.time()
distance_matrix, start_idx = get_personal_D(user_info, int(user_id), height_idx, delta_h_tolerance, delta_w_tolerance)
print("걸린시간 : ", round(time.time() - START, 2), "초")


# In[8]:


print(start_idx)                         


# In[9]:


"""#b = 0
for i in distance_matrix.index:
    a = 0
    for j in distance_matrix.columns:
        if isNaN(distance_matrix.loc[i, j]) == False:
            a += 1
    print("사용 될 수 있는 user 수 : ",a, "명")
#    b += a
#print("b",b)"""


# In[206]:


user_profile_info.loc[user_id] = get_user_profile(user_id, distance_matrix, start_idx, rating_info, clothing_info, size_category).iloc[0]


# In[209]:


user_profile_info.loc[user_id]['기장'].loc[:, 'numerator' : 'num']
user_profile_info.loc[user_id]['어깨'].loc[:, 'numerator' : 'num']
user_profile_info.loc[user_id]['가슴'].loc[:, 'numerator' : 'num']
user_profile_info.loc[user_id]['소매'].loc[:, 'numerator' : 'num']


# In[207]:


adjusted_user_profile_info.loc[user_id] = get_adjusted_user_profile(user_profile_info.loc[user_id])


# In[208]:


adjusted_user_profile_info.loc[user_id]['기장']
adjusted_user_profile_info.loc[user_id]['어깨']
adjusted_user_profile_info.loc[user_id]['가슴']
adjusted_user_profile_info.loc[user_id]['소매']


# In[180]:


test = get_compatibility_by_profile(user_id, '100015', adjusted_user_profile_info, clothing_info, size_category)


# In[179]:


test


# In[168]:


a, b = recommend_by_user_profile(user_id, ['100024', '100044', '100066'], adjusted_user_profile_info, clothing_info, size_category)


# In[169]:


a


# In[170]:


b


# In[171]:


b.loc['100024', 'compatibility']


# In[145]:


b.loc['100044', 'compatibility']


# In[146]:


b.loc['100066', 'compatibility']


# In[154]:


START = time.time()
user_size_info.loc[int(user_id)], log = get_estimate_size(int(user_id), distance_matrix, start_idx, rating_info, clothing_info, size_category)
print("추정에 사용된 rating 수 : ", log.index.size, "개")
print("걸린시간 : ", round(time.time() - START, 2), "초")


# In[155]:


log.sort_values(by = 'distance')


# In[156]:


user_info.loc[int(user_id)]


# In[157]:


for user in log.loc[:,'user_id']:
    print(user_info.loc[user])


# In[158]:


recommend = recommed_by_distance(user_size_info.loc[int(user_id)], ['100024', '100044', '100066'], clothing_info, size_category)


# In[159]:


recommend = recommend_by_cosine(user_size_info.loc[int(user_id)], ['100024', '100044', '100066'], clothing_info, size_category)


# In[160]:


error_value = get_error_and_evaluation(int(user_id), user_size_info.loc[int(user_id)], rating_info, clothing_info, size_category)
error_value


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


"""h_limit = 0
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
        
"""


# In[ ]:




