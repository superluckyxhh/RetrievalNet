import os
import random

file_path = '/home/user/dataset/gld2021/train_all_list.txt'
train_split_path = '/home/user/dataset/gld2021/train_split.txt'
test_split_path = '/home/user/dataset/gld2021/test_split.txt'

# TEST
# file_path = '/home/user/code/RetrievalNet/all.txt'
# train_split_path = '/home/user/code/RetrievalNet/train_split.txt'
# test_split_path = '/home/user/code/RetrievalNet/test_split.txt'

test_sample_rate = 0.2
  
  
with open(file_path, 'r') as f:
    raw = f.readlines()
    
length = len(raw)
test_split_nums = int(length * test_sample_rate)
test_sample = random.sample(raw, test_split_nums)

with open(test_split_path, 'w') as f:
    for test_line in test_sample:
        f.write(test_line)
        
print(f'Test split file write done {test_split_nums} images')

train_sample = list(set(raw).difference(set(test_sample)))
with open(train_split_path, 'w') as f:
    for train_line in train_sample:
        f.write(train_line) 

print(f'Train split file write done {len(train_sample)} images')