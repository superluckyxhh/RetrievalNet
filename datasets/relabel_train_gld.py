import os
from tqdm import tqdm
import glob


def get_test_images_files_and_labels(csv_path, image_dir, save_path):
    with open(csv_path, 'r') as f:
        raw = f.readlines()[1:]
    
    images = {}
    for line in raw:
        line = line.strip('\n')
        file_id, label = line.split(',')
        images[file_id] = {}
        images[file_id]['file_id'] = file_id
        images[file_id]['label'] = label
    
    image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))
    for image_path in tqdm(image_paths):
        file_id = os.path.basename(image_path)[:-4]
        if file_id in images:
            images[file_id]['image_path'] = image_path
    
    image_paths = []
    file_ids = []
    labels = []
    for _, value in images.items():
        image_paths.append(value['image_path'])
        file_ids.append(value['file_id'])
        labels.append(int(value['label']))
    
    unique_labels = sorted(set(labels))
    relabeling = {label: index for index, label in enumerate(unique_labels)}
    new_labels = [relabeling[label] for label in labels]
    
    with open(save_path, 'w') as f:
        for i in range(len(image_paths)):
            _info = image_paths[i] + ' ' + str(new_labels[i]) + '\n'
            f.write(_info)
    
    return image_paths, file_ids, new_labels, relabeling
   

# if __name__ == '__main__':
#     csv_path = '/home/user/dataset/gld2021/train.csv'
#     test_image_dir = '/home/user/dataset/gld2021/train/*/*/*/'
#     save_path = '/home/user/dataset/gld2021/train_relabel_list.txt'
#     image_paths, _, labels, _ = get_test_images_files_and_labels(csv_path, test_image_dir, save_path)
#     print('image nums: ', len(image_paths))
#     print('unique labels num: ', len(set(labels)))
#     print("Done!")