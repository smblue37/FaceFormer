import os
import torch
from collections import defaultdict
from torch.utils import data
import copy
import numpy as np
import pickle
from tqdm import tqdm
import random,math
from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Processor
import librosa   
import glob 

training_ids = ['M003', 'M005', 'M007', 'M009', 'M011', 'M012', 'M013', 'M019', 
                'M022', 'M023', 'M024', 'M025', 'M026', 'M027', 'M028', 'M029', 
                'M030', 'M031', 'W009', 'W011', 'W014', 'W015', 'W016', 'W018', 
                'W019', 'W021', 'W023', 'W024', 'W025', 'W026', 'W028', 'W029'
                ] # 32 ids
val_ids = ['M032', 'M033', 'M034', 'M035', 'W033', 'W035', 'W036']  # 7 ids

test_ids = ['M037', 'M039', 'M040', 'M041', 'M042', 'W037', 'W038', 'W040'] # 8 ids

MEAD_ACTOR_DICT = {'M003': 0, 'M005': 1, 'M007': 2, 'M009': 3, 'M011': 4, 'M012': 5, 'M013': 6, 'M019': 7, 
                   'M022': 8, 'M023': 9, 'M024': 10, 'M025': 11, 'M026': 12, 'M027': 13, 'M028': 14, 'M029': 15, 
                   'M030': 16, 'M031': 17, 'W009': 18, 'W011': 19, 'W014': 20, 'W015': 21, 'W016': 22, 'W018': 23, 
                   'W019': 24, 'W021': 25, 'W023': 26, 'W024': 27, 'W025': 28, 'W026': 29, 'W028': 30, 'W029': 31, # 32 train_ids
                   'M032': 32, 'M033': 33, 'M034': 34, 'M035': 35, 'W033': 36, 'W035': 37, 'W036': 38, # 7 val_ids
                   'M037': 39, 'M039': 40, 'M040': 41, 'M041': 42, 'M042': 43, 'W037': 44, 'W038': 45, 'W040': 46} # 8 test_ids

# EMOTION_DICT = {'neutral': 1, 'calm': 2, 'happy': 3, 'sad': 4, 'angry' :  5, 'fear': 6, 'disgusted': 7, 'surprised': 8, 'contempt' : 9}
# calm for RAVDESS
EMOTION_DICT = {'neutral': 1, 'happy': 2, 'sad': 3, 'surprised': 4, 'fear': 5, 'disgusted': 6, 'angry': 7, 'contempt': 8, 'calm' : 9}
# modify DICT to match inferno's original emotion label
modify_DICT = {1:1, 3:2, 4:3, 5:7, 6:5, 7:6, 8:4, 9:8}
GENDER_DICT = {'M' : 0, 'W' : 1}

class Dataset(data.Dataset):
    def __init__(self, data,subjects_dict,data_type="train"):
        self.data = data
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        #self.one_hot_labels = np.eye(len(subjects_dict["train"]))
        self.one_hot_labels = np.eye(len(subjects_dict["train"]))

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        file_name = self.data[index]["name"]
        #print(file_name)
        audio = self.data[index]["audio"]
        #print("audio")
        #print(audio.shape)
        vertice = self.data[index]["vertice"]
        print("vertice")
        print(vertice.shape)
        template = self.data[index]["template"]
        #print("template")
        #print(template.shape)
        if self.data_type == "train":
            #subject = "_".join(file_name.split("_")[:-1])
            subject = file_name.split("_")[0]
            one_hot = self.one_hot_labels[self.subjects_dict["train"].index(subject)]
            #check shape of one_hot
            #one_hot = one_hot.reshape((1,len(self.subjects_dict["train"])))
        else:
            one_hot = self.one_hot_labels
        #print(one_hot)
        #print(one_hot.shape)
            
        return torch.FloatTensor(audio),torch.FloatTensor(vertice), torch.FloatTensor(template), torch.FloatTensor(one_hot), file_name

    def __len__(self):
        return self.len

def read_data(args):
    print("Loading data...")
    data = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []
    
    audio_path = args.wav_path
    vertices_path = args.vertices_path

    template_file = os.path.join(args.dataset, args.template_file)
    print(template_file)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin,encoding='latin1')
        print("here")
    
    
    if args.dataset == 'MEAD':        
      # training set
      train_file_paths = []
      for training_id in training_ids:
        train_file_paths += glob.glob(os.path.join(args.vertices_path, training_id, '*.npy'))
        
      for file_path in tqdm(train_file_paths):
        uid = file_path.split('/')[-1].split('.')[0]
        actor_name = uid.split('_')[0] # M005
        # if args.split == 'train' :
        #     actor_id = MEAD_ACTOR_DICT[actor_name] # name -> id
        # ########################## set 01 for all val dataset
        # else :
        #     actor_id = 1
        # # actor_id = 1
        # # modify to match inferno emotion label
        # # emotion = int(uid.split('_')[1])
        # emotion = modify_DICT[int(uid.split('_')[1])]
        # intensity = int(uid.split("_")[2])
        # gender = GENDER_DICT[uid.split('_')[0][0]] # M -> 0, W -> 1
        
        audio_sample_path = file_path.replace("vertices", "audio_sample")
        if os.path.exists(audio_sample_path):
            data[uid]["name"] = file_path.split('/')[-1]
            #data[uid]["vertice"] = np.load(file_path, allow_pickle=True)
            data[uid]["vertice"] = np.load(file_path, allow_pickle=True).reshape((-1,5023*3))
            #temp_calcul = np.prod(np.load(file_path, allow_pickle=True).shape) // 15069
            #data[uid]["vertice"] = np.load(file_path, allow_pickle=True).reshape((temp_calcul,15069))
            data[uid]["audio"] = np.load(file_path.replace("vertices", "audio_sample"))
            #data[uid]["audio"] = np.load(file_path.replace("vertices", "audio_sample")).reshape((1,-1))
            data[uid]["template"] = templates.reshape((-1))
            #data[uid]["template"] = (templates.reshape((-1))).reshape((1,-1))
            
            train_data.append(data[uid])
        
      # val set
      val_file_paths = []
      for val_id in val_ids:
        val_file_paths += glob.glob(os.path.join(args.vertices_path, val_id, '*.npy'))
        
      for file_path in tqdm(val_file_paths):
        uid = file_path.split('/')[-1].split('.')[0]
        actor_name = uid.split('_')[0] # M005
        # if args.split == 'train' :
        #     actor_id = MEAD_ACTOR_DICT[actor_name] # name -> id
        # ########################## set 01 for all val dataset
        # else :
        #     actor_id = 1
        # # actor_id = 1
        # # modify to match inferno emotion label
        # # emotion = int(uid.split('_')[1])
        # emotion = modify_DICT[int(uid.split('_')[1])]
        # intensity = int(uid.split("_")[2])
        # gender = GENDER_DICT[uid.split('_')[0][0]] # M -> 0, W -> 1
        
        audio_sample_path = file_path.replace("vertices", "audio_sample")
        if os.path.exists(audio_sample_path):
            data[uid]["name"] = file_path.split('/')[-1]
            #data[uid]["vertice"] = np.load(file_path, allow_pickle=True)
            data[uid]["vertice"] = np.load(file_path, allow_pickle=True).reshape((-1,5023*3))
            #temp_calcul = np.prod(np.load(file_path, allow_pickle=True).shape) // 15069
            #data[uid]["vertice"] = np.load(file_path, allow_pickle=True).reshape((temp_calcul,15069))
            data[uid]["audio"] = np.load(file_path.replace("vertices", "audio_sample"))
            #data[uid]["audio"] = np.load(file_path.replace("vertices", "audio_sample")).reshape((1,-1))
            data[uid]["template"] = templates.reshape((-1))
            #data[uid]["template"] = (templates.reshape((-1))).reshape((1,-1))
            
            valid_data.append(data[uid])
        
      # test set
      test_file_paths = []
      for test_id in test_ids:
        test_file_paths += glob.glob(os.path.join(args.vertices_path, test_id, '*.npy'))
        
      for file_path in tqdm(test_file_paths):
        uid = file_path.split('/')[-1].split('.')[0]
        actor_name = uid.split('_')[0] # M005
        # if args.split == 'train' :
        #     actor_id = MEAD_ACTOR_DICT[actor_name] # name -> id
        # ########################## set 01 for all val dataset
        # else :
        #     actor_id = 1
        # # actor_id = 1
        # # modify to match inferno emotion label
        # # emotion = int(uid.split('_')[1])
        # emotion = modify_DICT[int(uid.split('_')[1])]
        # intensity = int(uid.split("_")[2])
        # gender = GENDER_DICT[uid.split('_')[0][0]] # M -> 0, W -> 1
        
        audio_sample_path = file_path.replace("vertices", "audio_sample")
        if os.path.exists(audio_sample_path):
            data[uid]["name"] = file_path.split('/')[-1]
            #data[uid]["vertice"] = np.load(file_path, allow_pickle=True)
            data[uid]["vertice"] = np.load(file_path, allow_pickle=True).reshape((-1,5023*3))
            #temp_calcul = np.prod(np.load(file_path, allow_pickle=True).shape) // 15069
            #data[uid]["vertice"] = np.load(file_path, allow_pickle=True).reshape((temp_calcul,15069))
            data[uid]["audio"] = np.load(file_path.replace("vertices", "audio_sample"))
            #data[uid]["audio"] = np.load(file_path.replace("vertices", "audio_sample")).reshape((1,-1))
            data[uid]["template"] = templates.reshape((-1))
            #data[uid]["template"] = (templates.reshape((-1))).reshape((1,-1))
            
            test_data.append(data[uid])

      subjects_dict = {}
      #subjects_dict["train"] = [i for i in args.train_subjects.split(" ")]
      #subjects_dict["val"] = [i for i in args.val_subjects.split(" ")]
      #subjects_dict["test"] = [i for i in args.test_subjects.split(" ")]
      subjects_dict["train"] = training_ids
      subjects_dict["val"] = val_ids
      subjects_dict["test"] = test_ids

    splits = {'vocaset':{'train':range(1,41),'val':range(21,41),'test':range(21,41)},
     'BIWI':{'train':range(1,33),'val':range(33,37),'test':range(37,41)}}

    print(len(train_data), len(valid_data), len(test_data))
    return train_data, valid_data, test_data, subjects_dict

def get_dataloaders(args):
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data(args)
    train_data = Dataset(train_data,subjects_dict,"train")
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    print(dataset["train"])
    valid_data = Dataset(valid_data,subjects_dict,"val")
    dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=False)
    test_data = Dataset(test_data,subjects_dict,"test")
    dataset["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    
    #dataset_path = "/home/smsm0307/FaceFormer-main/dataset.pkl"
    #with open(dataset_path, 'wb') as file:
    #    pickle.dump(dataset, file)
    
    print("finally...")
    return dataset

if __name__ == "__main__":
    get_dataloaders()