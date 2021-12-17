from torch.utils.data import Dataset
import numpy as np
import os
import torch
from torchvision.transforms import Resize,CenterCrop
import torchvision.transforms as transform
import pandas as pd
from PIL import Image

class RADIal(Dataset):

    def __init__(self, root_dir,difficult=False):

        self.root_dir = root_dir
        
        self.labels = pd.read_csv(os.path.join(root_dir,'labels.csv')).to_numpy()
       
        # Keeps only easy samples
        if(difficult==False):
            ids_filters=[]
            ids = np.where( self.labels[:, -1] == 0)[0]
            ids_filters.append(ids)
            ids_filters = np.unique(np.concatenate(ids_filters))
            self.labels = self.labels[ids_filters]


        # Gather each input entries by their sample id
        self.unique_ids = np.unique(self.labels[:,0])
        self.label_dict = {}
        for i,ids in enumerate(self.unique_ids):
            sample_ids = np.where(self.labels[:,0]==ids)[0]
            self.label_dict[ids]=sample_ids
        self.sample_keys = list(self.label_dict.keys())
        

        self.resize = Resize((256,224), interpolation=transform.InterpolationMode.NEAREST)
        self.crop = CenterCrop((512,448))


    def __len__(self):
        return len(self.label_dict)

    def __getitem__(self, index):
        
        # Get the sample id
        sample_id = self.sample_keys[index] 

        # From the sample id, retrieve all the labels ids
        entries_indexes = self.label_dict[sample_id]

        # Get the objects labels
        box_labels = self.labels[entries_indexes]

        # Labels contains following parameters:
        # numSample	x1_pix	y1_pix	x2_pix	y2_pix	laser_X_m	laser_Y_m	laser_Z_m radar_X_m	radar_Y_m	radar_R_m
        # radar_A_deg	radar_D_mps	radar_P_db
        box_labels = box_labels[:,1:-3].astype(np.float32)

        # Read the Radar FFT data
        radar_name = os.path.join(self.root_dir,'radar_FFT',"fft_{:06d}.npz".format(sample_id))
        input = np.load(radar_name)['arr_0']
        radar_FFT = np.concatenate([input.real,input.imag],axis=2)
       
        # Read the Radar point cloud
        filename = os.path.join(self.root_dir,'radar_PCL',"pcl_{:06d}.npy".format(sample_id))
        # range,azimuth,elevation,power,doppler,x,y,z,v
        radar_pc = np.load(filename,allow_pickle=True)[[5,6,7],:]   # Keeps only x,y,z
        radar_pc = np.rollaxis(radar_pc,1,0)
        radar_pc[:,1] *= -1

        # Read the Laser point cloud
        filename = os.path.join(self.root_dir,'laser_PCL',"pcl_{:06d}.npy".format(sample_id))
        #    float x,y,z;
        #    float intensity;
        #    float radialDistance;
        #    float polAngle,aziAngle;
        #    uint8_t layer_index;
        laser_pc = np.load(filename,allow_pickle=True)[:,:3]    # Keeps only x,y,z

    
        # Read the segmentation map
        segmap_name = os.path.join(self.root_dir,'radar_Freespace',"freespace_{:06d}.png".format(sample_id))
        segmap = Image.open(segmap_name) # [512,900]
        # 512 pix forthe range and 900 pix for the horizontal FOV (180deg)
        # We crop the fov to 89.6deg
        segmap = self.crop(segmap)
        # and we resize to half of its size
        segmap = np.asarray(self.resize(segmap))

        # Read the camera image
        img_name = os.path.join(self.root_dir,'camera',"image_{:06d}.jpg".format(sample_id))
        image = np.asarray(Image.open(img_name))

        return image, radar_FFT, radar_pc, laser_pc,segmap,box_labels
