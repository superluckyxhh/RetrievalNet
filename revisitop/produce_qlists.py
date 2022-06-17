import os
from dataset import configdataset


# test_dataset = 'roxford5k'
test_dataset = 'rparis6k'
data_root = '/home/user/dataset/data'

save_dir = '/home/user/code/RetrievalNet/revisitop'
query_save_path = os.path.join(save_dir, 'query_' + test_dataset + '.txt')
database_save_path = os.path.join(save_dir, 'database_' + test_dataset + '.txt')

cfg = configdataset(test_dataset, os.path.join(data_root, 'datasets'))

query_lists = cfg["qimlist"]
database_lists = cfg['imlist']

# Save query.txt
with open(query_save_path, 'w') as f:
    for i in range(len(query_lists)):
        query_name = query_lists[i]
        f.write(query_name + '\n')
print("Query Images Written Done")

# Save database.txt    
with open(database_save_path, 'w') as f:
    for i in range(len(database_lists)):
        db_name = database_lists[i]
        f.write(db_name + '\n')
print("Databse Images Written Done")