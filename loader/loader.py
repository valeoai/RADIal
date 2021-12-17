import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import torch

Sequences = {'Validation':['RECORD@2020-11-22_12.49.56','RECORD@2020-11-22_12.11.49','RECORD@2020-11-22_12.28.47','RECORD@2020-11-21_14.25.06'],
            'Test':['RECORD@2020-11-22_12.45.05','RECORD@2020-11-22_12.25.47','RECORD@2020-11-22_12.03.47','RECORD@2020-11-22_12.54.38']}

def RADIal_collate(batch):
    images = []
    FFTs = []
    laser_pcs = []
    radar_pcs = []
    segmaps = []
    labels = []

    for image, radar_FFT, radar_pc, laser_pc,segmap,box_labels in batch:
        labels.append(torch.from_numpy(box_labels))
        images.append(torch.tensor(image))
        FFTs.append(torch.tensor(radar_FFT).permute(2,0,1))
        segmaps.append(torch.tensor(segmap))
        laser_pcs.append(torch.from_numpy(laser_pc))
        radar_pcs.append(torch.from_numpy(radar_pc))
        
    return torch.stack(images), torch.stack(FFTs), torch.stack(segmaps),laser_pcs,radar_pcs,labels

def CreateDataLoaders(dataset,batch_size=4,num_workers=2,seed=0):

    dict_index_to_keys = {s:i for i,s in enumerate(dataset.sample_keys)}

    Val_indexes = []
    for seq in Sequences['Validation']:
        idx = np.where(dataset.labels[:,14]==seq)[0]
        Val_indexes.append(dataset.labels[idx,0])
    Val_indexes = np.unique(np.concatenate(Val_indexes))

    Test_indexes = []
    for seq in Sequences['Test']:
        idx = np.where(dataset.labels[:,14]==seq)[0]
        Test_indexes.append(dataset.labels[idx,0])
    Test_indexes = np.unique(np.concatenate(Test_indexes))

    val_ids = [dict_index_to_keys[k] for k in Val_indexes]
    test_ids = [dict_index_to_keys[k] for k in Test_indexes]
    train_ids = np.setdiff1d(np.arange(len(dataset)),np.concatenate([val_ids,test_ids]))

    train_dataset = Subset(dataset,train_ids)
    val_dataset = Subset(dataset,val_ids)
    test_dataset = Subset(dataset,test_ids)

    # create data_loaders
    train_loader = DataLoader(train_dataset, 
                            batch_size=batch_size, 
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=True,
                            collate_fn=RADIal_collate)
    val_loader =  DataLoader(val_dataset, 
                            batch_size=batch_size, 
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True,
                            collate_fn=RADIal_collate)
    test_loader =  DataLoader(test_dataset, 
                            batch_size=batch_size, 
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True,
                            collate_fn=RADIal_collate)

    return train_loader,val_loader,test_loader