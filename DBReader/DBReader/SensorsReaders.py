import numpy as np
import sys
import os
import cv2
import struct
import cantools

class CameraReader():
    def __init__(self,dict):
        self.dict = dict

        # read video
        if(np.shape(self.dict['offset'])[0]>0):
            self.f = open(str(self.dict['filename']),'rb')
        else:
            self.f = cv2.VideoCapture(str(self.dict['filename']))
            
        self.ImageWidth = 1920
        self.ImageHeight = 1080
        
    def GetData(self,index):

        # MJPG mode
        if(np.shape(self.dict['offset'])[0]>0):
            offset = int(self.dict['offset'][index])
            length = int(self.dict['datasize'][index])
            self.f.seek(offset)
            data = self.f.read(length)
            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        # AVI MOde
        else:
            self.f.set(cv2.CAP_PROP_POS_FRAMES,index)
            ret, frame = self.f.read()

        return frame
    
    def GetTimestamp(self,index):
        return self.dict['timestamp'][index]
    def GetTimeOfIssue(self,index):
        return self.dict['timeofissue'][index]
    def GetOffet(self,index):
        return int(self.dict['offset'][index])

    def GetSampleNumber(self,index):
        return int(self.dict['sample'][index])

class CANDecoder():
    def __init__(self,dbc):
        self.db = cantools.database.load_file(dbc)
 
    def decode(self,messages):
        Signals=[]
        for i,m in enumerate(messages):
            Signals.append({'signals':self.db.decode_message(m['ID'],m['DATA']),'timestamp':m['timestamp']})
            
        return Signals


class CANReader():
    def __init__(self,dict):
        self.dict = dict
        self.fd = open(str(self.dict['filename']),'rb')
        struct_fmt = '=QI8B4B'
        self.struct_len = struct.calcsize(struct_fmt)
 
    def GetData(self,index):
        index=int(index)
        offset = self.dict['offset'][index]
        datasize = self.dict['datasize'][index]
        
        self.fd.seek(offset)
        messages=[]
        for i in range(int(datasize/self.struct_len)):
            raw_data = self.fd.read(self.struct_len)
            if(len(raw_data)==self.struct_len):
            	messages.append({'timestamp':struct.unpack_from('Q', raw_data, 0)[0],
                             'ID':struct.unpack_from('I', raw_data, 8)[0],
                             'DATA':raw_data[12:20]})
      
        return messages
    
    def GetTimestamp(self,index):
        return self.dict['timestamp'][index]
    
    def GetTimeOfIssue(self,index):
        return self.dict['timeofissue'][index]

    def GetOffet(self,index):
        return int(self.dict['offset'][index])

    def GetSampleNumber(self,index):
        return int(self.dict['sample'][index])

class LaserReader():
    def __init__(self,dict):
        self.dict = dict
        self.fd = open(str(self.dict['filename']),'rb')
        
        struct_fmt = '=7f4B'
        self.struct_len = struct.calcsize(struct_fmt)
        self.struct_unpack = struct.Struct(struct_fmt).unpack_from

    def GetData(self,index):
        index=int(index)
        offset = self.dict['offset'][index]
        datasize = self.dict['datasize'][index]
        
        self.fd.seek(offset)
        pts3d=[]
        for i in range(int(datasize/self.struct_len)):
            pts3d.append(self.struct_unpack(self.fd.read(self.struct_len)))

        pts3d = np.asarray(pts3d)
       
        return pts3d
    
    def GetTimestamp(self,index):
        return self.dict['timestamp'][index]

    def GetTimeOfIssue(self,index):
        return self.dict['timeofissue'][index]

    def GetOffet(self,index):
        return int(self.dict['offset'][index])
    
    def GetSampleNumber(self,index):
        return int(self.dict['sample'][index])

class GPSReader():
    def __init__(self,dict):
        self.dict = dict
        fd = open(str(self.dict['filename']),'r')
        self.data = fd.readlines()
    def GetData(self,index):
        return self.data[index]
    def GetTimestamp(self,index):
        return self.dict['timestamp'][index]

    def GetTimeOfIssue(self,index):
        return self.dict['timeofissue'][index]
    

class RadarReader():
    def __init__(self,dict):
        self.dict = dict
        self.fd = open(str(self.dict['filename']),'rb')
        
    def GetData(self,index):
        offset = int(self.dict['offset'][index])
        datasize = self.dict['datasize'][index]
        self.fd.seek(offset)
        data = np.fromfile(self.fd, dtype=np.int16,count=int(datasize/2))

        return data  
    
    def GetTimestamp(self,index):
        return self.dict['timestamp'][index]

    def GetTimeOfIssue(self,index):
        return self.dict['timeofissue'][index]

    def GetOffet(self,index):
        return int(self.dict['offset'][index])

    def GetSampleNumber(self,index):
        return int(self.dict['sample'][index])