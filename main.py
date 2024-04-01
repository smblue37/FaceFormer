import re, random, math
import numpy as np
import argparse
from tqdm import tqdm
import os, shutil
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import get_dataloaders_pkl
from data_loader import get_dataloaders
from faceformer import Faceformer

import wandb
import pickle

training_ids = ['M003', 'M005', 'M007', 'M009', 'M011', 'M012', 'M013', 'M019', 
                'M022', 'M023', 'M024', 'M025', 'M026', 'M027', 'M028', 'M029', 
                'M030', 'M031', 'W009', 'W011', 'W014', 'W015', 'W016', 'W018', 
                'W019', 'W021', 'W023', 'W024', 'W025', 'W026', 'W028', 'W029'
                ]
device = torch.device("cuda")

def trainer(args, train_loader, dev_loader, model, optimizer, criterion, epoch=100):
    save_path = os.path.join(args.dataset,args.save_path)
    if os.path.exists(save_path):
        #shutil.rmtree(save_path)
        print("exist")
    else:
      os.makedirs(save_path)

    train_subjects_list = training_ids
    iteration = 0
    for e in range(epoch+1):
        loss_log = []
        # train
        model.train()
        pbar = tqdm(enumerate(train_loader),total=len(train_loader))
        optimizer.zero_grad()

        for i, (audio, vertice, template, one_hot, file_name) in pbar:
            iteration += 1
            if i < len(pbar) - 1:
              # to gpu
              audio, vertice, template, one_hot  = audio.to(device), vertice.to(device), template.to(device), one_hot.to(device)
              loss = model(audio, template, vertice, one_hot, criterion, teacher_forcing=False)
              loss.backward()
              loss_log.append(loss.item())
              if i % args.gradient_accumulation_steps==0:
                  optimizer.step()
                  optimizer.zero_grad()

              torch.autograd.set_detect_anomaly(True)
              pbar.set_description("(Epoch {}, iteration {}) TRAIN LOSS:{:.7f}".format((e+1), iteration ,np.mean(loss_log)))
              wandb.log({"Epoch": (e+1), "Train_Loss": np.mean(loss_log)})
        # validation
        valid_loss_log = []
        model.eval()
        for audio, vertice, template, one_hot_all,file_name in dev_loader:
            # to gpu
            audio, vertice, template, one_hot_all= audio.to(device), vertice.to(device), template.to(device), one_hot_all.to(device)
            train_subject = "_".join(file_name[0].split("_")[:-1])
            if train_subject in train_subjects_list:
                condition_subject = train_subject
                iter = train_subjects_list.index(condition_subject)
                one_hot = one_hot_all[:,iter,:]
                loss = model(audio, template, vertice, one_hot, criterion)
                valid_loss_log.append(loss.item())
            else:
                for iter in range(one_hot_all.shape[-1]):
                    condition_subject = train_subjects_list[iter]
                    one_hot = one_hot_all[:,iter,:]
                    loss = model(audio, template, vertice, one_hot, criterion)
                    valid_loss_log.append(loss.item())
                        
            wandb.log({"Valid_Loss": np.mean(valid_loss_log)})
                        
        current_loss = np.mean(valid_loss_log)
        
        #if (e > 0 and e % 25 == 0) or e == args.max_epoch:
        #if (e > 0 and e % epoch == 0) or e == args.max_epoch:
        torch.save(model.state_dict(), os.path.join(save_path,'{}_model.pth'.format(e)))

        print("epcoh: {}, current loss:{:.7f}".format(e+1,current_loss))    
    return model

@torch.no_grad()
def test(args, model, test_loader,epoch):
    result_path = os.path.join(args.dataset,args.result_path)
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path)

    save_path = os.path.join(args.dataset,args.save_path)
    train_subjects_list = training_ids

    model.load_state_dict(torch.load(os.path.join(save_path, '{}_model.pth'.format(epoch))))
    model = model.to(device)
    model.eval()
   
    for audio, vertice, template, one_hot_all, file_name in test_loader:
        # to gpu
        audio, vertice, template, one_hot_all= audio.to(device), vertice.to(device), template.to(device), one_hot_all.to(device)
        train_subject = file_name[0].split("_")[0]
        if train_subject in train_subjects_list:
            condition_subject = train_subject
            iter = train_subjects_list.index(condition_subject)
            one_hot = one_hot_all[:,iter,:]
            prediction = model.predict(audio, template, one_hot)
            prediction = prediction.squeeze() # (seq_len, V*3)
            np.save(os.path.join(result_path, file_name[0].split(".")[0]+"_condition_"+condition_subject+".npy"), prediction.detach().cpu().numpy())
        else:
            for iter in range(one_hot_all.shape[-1]):
                condition_subject = train_subjects_list[iter]
                one_hot = one_hot_all[:,iter,:]
                prediction = model.predict(audio, template, one_hot)
                prediction = prediction.squeeze() # (seq_len, V*3)
                np.save(os.path.join(result_path, file_name[0].split(".")[0]+"_condition_"+condition_subject+".npy"), prediction.detach().cpu().numpy())
         
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser(description='FaceFormer: Speech-Driven 3D Facial Animation with Transformers')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--dataset", type=str, default="MEAD", help='MEAD')
    parser.add_argument("--vertice_dim", type=int, default=5023*3, help='number of vertices - 5023*3 for vocaset; 23370*3 for BIWI')
    parser.add_argument("--feature_dim", type=int, default=64, help='64 for vocaset; 128 for BIWI')
    parser.add_argument("--period", type=int, default=25, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--wav_path", type=str, default= "wav", help='path of the audio signals')
    parser.add_argument("--vertices_path", type=str, default="vertices_npy", help='path of the ground truth')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, default=100, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--template_file", type=str, default="templates.pkl", help='path of the personalized templates')
    parser.add_argument("--save_path", type=str, default="save", help='path of the trained models')
    parser.add_argument("--result_path", type=str, default="result", help='path to the predictions')
    parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA"
       " FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA"
       " FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA"
       " FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
    parser.add_argument("--val_subjects", type=str, default="FaceTalk_170811_03275_TA"
       " FaceTalk_170908_03277_TA")
    parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA"
       " FaceTalk_170731_00024_TA")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    wandb.init(project='FaceFormer', entity='mesh_talk')
    wandb.config = {"learning_rate: 0.001"}

    #build model
    model = Faceformer(args)
    print("model parameters: ", count_parameters(model))

    # to cuda
    assert torch.cuda.is_available()
    model = model.to(device)
    
    device_count = torch.cuda.device_count()
    
    model = model.to(device)
    
    #load data
    #dataset = get_dataloaders(args)
    
    # if dataset is already loaded
    if args.batch_size == 1:
      train_data_path = "/scratch/smsm0307/dataset/train_data.pkl"
      if os.path.exists(train_data_path):
          with open(train_data_path, 'rb') as file:
            train_data = pickle.load(file)
      else:
          with open(train_data_path, 'wb') as file:
            pickle.dump(train_data, file)
      
      valid_data_path = "/scratch/smsm0307/dataset/valid_data.pkl"
      if os.path.exists(valid_data_path):
          with open(valid_data_path, 'rb') as file:
            valid_data = pickle.load(file)
      else:
          with open(valid_data_path, 'wb') as file:
            pickle.dump(valid_data, file)
     
      test_data_path = "/scratch/smsm0307/dataset/test_data.pkl"
      if os.path.exists(test_data_path):
          with open(test_data_path, 'rb') as file:
            test_data = pickle.load(file)
      else:
          with open(test_data_path, 'wb') as file:
            pickle.dump(test_data, file)
            
    elif args.batch_size >= 2:
      train_data_path = "/scratch/smsm0307/dataset/2sec_train_data.pkl"
      if os.path.exists(train_data_path):
        with open(train_data_path, 'rb') as file:
          train_data = pickle.load(file)
      else:
        with open(train_data_path, 'wb') as file:
          pickle.dump(train_data, file)
      
      valid_data_path = "/scratch/smsm0307/dataset/2sec_valid_data.pkl"
      if os.path.exists(valid_data_path):
        with open(valid_data_path, 'rb') as file:
          valid_data = pickle.load(file)
      else:
        with open(valid_data_path, 'wb') as file:
          pickle.dump(valid_data, file)
      
      test_data_path = "/scratch/smsm0307/dataset/2sec_test_data.pkl"
      if os.path.exists(test_data_path):
        with open(test_data_path, 'rb') as file:
          test_data = pickle.load(file)
      else:
        with open(test_data_path, 'wb') as file:
          pickle.dump(test_data, file)
        
    dataset = get_dataloaders_pkl(train_data, valid_data, test_data, args.batch_size)
    
    # loss
    criterion = nn.MSELoss()

    # Train the model
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=args.lr)
    
    wandb.watch(model, criterion, log="all", log_freq=10)
    
    model = trainer(args, dataset["train"], dataset["valid"],model, optimizer, criterion, epoch=args.max_epoch)
    #model = trainer(args, dataset["train"], dataset["valid"],model, optimizer, criterion, epoch=10)
    
    #test(args, model, dataset["test"], epoch=args.max_epoch)
    wandb.finish()
    
if __name__=="__main__":
    main()