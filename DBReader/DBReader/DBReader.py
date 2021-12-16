import numpy as np
from pathlib import Path
from os import listdir
from os.path import isfile, join
import pandas as pd

from .SensorsReaders import CameraReader, CANReader, LaserReader, GPSReader, RadarReader, CANDecoder

offsetTable = {'camera':0,'scala':-40000,'radar_ch0':-180000,'radar_ch1':-180000,'radar_ch2':-180000,'radar_ch3':-180000,'gps':0,'can':0}

default_sensor_list = ['camera','scala','radar_ch0','radar_ch1','radar_ch2','radar_ch3','can']
def parse_recording(folder):
    recorder_folder_path = Path(folder)
    
    list_files = [file for file in listdir(recorder_folder_path) if isfile(join(recorder_folder_path, file))]
    
    # list the number of sensor recorded
    list_of_sensors = []
    list_files2 = []
    for file in list_files:
        sensor = file[len(recorder_folder_path.name)+1:]
        sensor = sensor[:sensor.rfind('.')]

        if(sensor in list(offsetTable.keys())):
            list_of_sensors.append(sensor)
            list_files2.append(file)

    list_files = list_files2
    dict_sensor = {s:{'filename':recorder_folder_path/list_files[i],'timestamp': [],'timeofissue': [],'sample': [], 'offset': [], 'datasize' :[]} for i,s in enumerate(list_of_sensors)}


    # 1. Open the REC file
    rec_file_name = recorder_folder_path.name+'_events_log.rec'
    rec_file_path = recorder_folder_path/rec_file_name
    
    df = pd.read_csv(rec_file_path,header=None)
    data = df.values
    
    for line in data:
        elt = line[0].split()
        timestamp = int(elt[1])
        timeofissue = int(elt[4])
        sample = int(elt[7])
        sensor = elt[10]

        if(sensor not in dict_sensor.keys()):
            continue
        
        dict_sensor[sensor]['timestamp'].append(timestamp)
        dict_sensor[sensor]['timeofissue'].append(timeofissue)
        dict_sensor[sensor]['sample'].append(sample)
        
        if(len(elt)==17):
            # timestamp - sample - sensor - offset - datasize
            dict_sensor[sensor]['offset'].append(int(elt[13]))
            dict_sensor[sensor]['datasize'].append(int(elt[16]))
            
    for sensor in dict_sensor:
        if(len(dict_sensor[sensor]['timeofissue'])>0 and len(dict_sensor[sensor]['timestamp'])>0):

            offset = dict_sensor[sensor]['timeofissue'][0] - dict_sensor[sensor]['timestamp'][0] + offsetTable[sensor]
            for i in range(len(dict_sensor[sensor]['timestamp'])):
                dict_sensor[sensor]['timestamp'][i] += offset

        dict_sensor[sensor]['timestamp'] = np.asarray(dict_sensor[sensor]['timestamp'])
        dict_sensor[sensor]['timeofissue'] = np.asarray(dict_sensor[sensor]['timeofissue'])
        dict_sensor[sensor]['sample'] = np.asarray(dict_sensor[sensor]['sample'])
        dict_sensor[sensor]['offset'] = np.asarray(dict_sensor[sensor]['offset'])
        dict_sensor[sensor]['datasize'] = np.asarray(dict_sensor[sensor]['datasize'])


    return dict_sensor
           
class SyncReader():
    def __init__(self,folder,master=None,tolerance=200000,sync_mode='timestamp',silent=False):
        self.recorder_folder_path = Path(folder)
        self.sync_mode = sync_mode
        self.silent = silent

        self.dicts = parse_recording(folder)
        
        # 1. Open the REC file
        self.rec_file_name = self.recorder_folder_path.name+'_events_log.rec'
        self.rec_file_path = self.recorder_folder_path/self.rec_file_name
        
        labls = np.array([str(i) for i in range(22)]) # create some row names
        df = pd.read_csv(self.rec_file_path,header=None,names=labls,sep='[-|\s+]',engine='python')
        nbColumn = df.shape[1]
        self.df = df.iloc[:, np.arange(1,nbColumn,4)]
        self.df.columns = ['timestamp', 'timeofissue','data_sample', 'sensor', 'offset','datasize']
        self.sensorsFilters = self.df['sensor'].unique()
        self.filters = []
        self.df_filtered = self.df
        
        if(not self.silent):
        	print('-------------------------------------------------------------------------')
        	print('- Sensors available:')
        	for s in self.dicts.keys():
            		print('-    ',s)
        	print('-')
        	print('- You might use function "setSensorFilters" to select sensors you want to read!')
        	print('-------------------------------------------------------------------------')
        

        self.readers={}
        for sensor in self.sensorsFilters:

            # Make sure the sensor file exist!
            if(sensor not in self.dicts.keys()):
                continue

            if(sensor=='camera'):
                self.readers[sensor] = CameraReader(self.dicts[sensor])
            if(sensor=='can'):
                self.readers[sensor] = CANReader(self.dicts[sensor])
            if(sensor=='radar_ch1'):
                self.readers[sensor] = RadarReader(self.dicts[sensor])
            if(sensor=='radar_ch2'):
                self.readers[sensor] = RadarReader(self.dicts[sensor])
            if(sensor=='radar_ch3'):
                self.readers[sensor] = RadarReader(self.dicts[sensor])         
            if(sensor=='radar_ch0'):
                self.readers[sensor] = RadarReader(self.dicts[sensor])                 
            if(sensor=='scala'):
                self.readers[sensor] = LaserReader(self.dicts[sensor])

        id_to_del = []
        nb_corrupted = 0
        nb_tolerance = 0
        
        # for Each radar sample, find the clostest sample for each sensor
        if(master is None):
            # by default, we use the Radar as Matser sensor
            if('radar_ch0' not in self.dicts or 'radar_ch1' not in self.dicts 
               or 'radar_ch2' not in self.dicts or 'radar_ch3' not in self.dicts):
                print('Error: recording does not contains the 4 radar chips')
            
            keys =list(self.dicts.keys())
            self.keys = keys
            
            if('gps' in self.dicts):
                keys.remove('gps')
            if('preview' in self.dicts):
                keys.remove('preview')
            if('None' in self.dicts):
                keys.remove('None')
            keys.remove('radar_ch0')
            keys.remove('radar_ch1')
            keys.remove('radar_ch2')
            keys.remove('radar_ch3')
            
            self.table=[]
            
            # Check the length of all radar recordings!
            NbSample = len(self.dicts['radar_ch3']['timestamp'])

            # Sequence is radar_ch3 radar_ch0 radar_ch2 radar_ch1
            for i in range(NbSample):
                timestamp = self.dicts['radar_ch3']['timestamp'][i]
                timeofissue = self.dicts['radar_ch3']['timeofissue'][i]
                FrameNumber = self.dicts['radar_ch3']['sample'][i]

                idx0 = np.where(self.dicts['radar_ch0']['sample']==(FrameNumber+1))[0]
                idx2 = np.where(self.dicts['radar_ch2']['sample']==(FrameNumber+2))[0]
                idx1 = np.where(self.dicts['radar_ch1']['sample']==(FrameNumber+3))[0]
                match={}

                match['radar_ch3'] = i

                if(len(idx0)==0 or len(idx1)==0 or len(idx2)==0):
                    id_to_del.append(i)
                    nb_corrupted+=1
                    match['radar_ch0'] = -1
                    match['radar_ch1'] = -1
                    match['radar_ch2'] = -1
                else:
                    match['radar_ch0'] = idx0[0]
                    match['radar_ch1'] = idx1[0]
                    match['radar_ch2'] = idx2[0]

                
                if(self.sync_mode=='timestamp'):
                    for k in keys:
                        if(len(self.dicts[k]['timestamp'])>0):
                            time_diff = np.abs(np.asarray(self.dicts[k]['timestamp']) - timestamp)
                            vmin = time_diff.min()
                            index_min = time_diff.argmin()

                            if(vmin>tolerance):
                                index_min=-1
                        else:
                            index_min=-1
                        
                        if(index_min==-1):
                            nb_tolerance+=1
                            id_to_del.append(i)

                        match[k] = index_min
                else:
                    for k in keys:
                        if(len(self.dicts[k]['timeofissue'])>0):
                            time_diff = np.abs(np.asarray(self.dicts[k]['timeofissue']) - timeofissue)
                            vmin = time_diff.min()
                            index_min = time_diff.argmin()
                            if(vmin>tolerance):
                                index_min=-1
                        else:
                            index_min=-1
                        
                        if(index_min==-1):
                            nb_tolerance+=1
                            id_to_del.append(i)

                        match[k] = index_min
            
                self.table.append(match)

            self.table = np.asarray(self.table)

            # Keep only sync samples
            id_to_del = np.unique(np.asarray(id_to_del))
            id_total = np.arange(len(self.table))
            self.id_valid = np.setdiff1d(id_total, id_to_del)
            if(not self.silent):
            	print('Total tolerance errors: ',nb_tolerance/len(self.table)*100,'%')
            	print('Total corrupted frames: ',nb_corrupted/len(self.table)*100,'%')
            self.table = self.table[self.id_valid]


        elif(master=='camera'):
            # we discard the radar, and consider only camera, laser, can
            if('camera' not in self.dicts):
                print('Error: recording does not contains camera')
            
            keys =list(self.dicts.keys())

            if('gps' in self.dicts):
                keys.remove('gps')
            if('radar_ch0' in self.dicts):
                keys.remove('radar_ch0')                
            if('radar_ch1' in self.dicts):
                keys.remove('radar_ch1')  
            if('radar_ch2' in self.dicts):
                keys.remove('radar_ch2')  
            if('radar_ch3' in self.dicts):
                keys.remove('radar_ch3')  
            if('preview' in self.dicts):
                keys.remove('preview')
            if('None' in self.dicts):
                keys.remove('None')

            self.keys = keys

            self.table=[]
            for i in range(len(self.dicts['camera']['timestamp'])):

                timestamp = self.dicts['camera']['timestamp'][i]  
                timeofissue = self.dicts['camera']['timeofissue'][i]


                match={}
                match['camera'] = i
                
                if(self.sync_mode=='timestamp'):
                    for k in keys:
                        if(len(self.dicts[k]['timestamp'])>0):
                            time_diff = np.abs(np.asarray(self.dicts[k]['timestamp']) - timestamp)
                            vmin = time_diff.min()
                            index_min = time_diff.argmin()
                        
                            if(vmin>tolerance):
                                nb_tolerance+=1
                                index_min = -1
                                id_to_del.append(i)
                        else:
                            index_min = -1
                        
                        match[k] = index_min
                else:
                    for k in keys:
                        if(len(self.dicts[k]['timeofissue'])>0):
                            time_diff = np.abs(np.asarray(self.dicts[k]['timeofissue']) - timeofissue)
                            vmin = time_diff.min()
                            index_min = time_diff.argmin()
                        
                            if(vmin>tolerance):
                                nb_tolerance+=1
                                id_to_del.append(i)
                                index_min = -1
                        else:
                            index_min = -1
                        
                        match[k] = index_min
            
                self.table.append(match)

            self.table = np.asarray(self.table)
            # Keep only sync samples
            id_to_del = np.unique(np.asarray(id_to_del))
            id_total = np.arange(len(self.table))
            id_to_keep = np.setdiff1d(id_total, id_to_del)
            if(not self.silent):
            	print('Total tolerance errors: ',nb_tolerance/len(self.table)*100,'%')
            self.table = self.table[id_to_keep]

        else:
            print('Mode not supported')
            return


        self.can_frames={'timestamp':[],'ID':[],'data':[]}
        if('can' in self.dicts):
            A = []
            for i in range(len(self.readers['can'].dict['offset'])):
                A.append(self.readers['can'].GetData(i))
            
            A=np.concatenate(A)
            
            for i in range(len(A)):
                self.can_frames['timestamp'].append(A[i]['timestamp'])
                self.can_frames['ID'].append(A[i]['ID'])
                self.can_frames['data'].append(A[i]['DATA'])
            self.can_frames['ID'] = np.asarray(self.can_frames['ID'])
            self.can_frames['timestamp'] = np.asarray(self.can_frames['timestamp'])
    

    def __len__(self):
        return len(self.table)
           
    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self.table):
            
            data={}
            for sensor in self.table[self.n]:
                index = self.table[self.n][sensor]
                data[sensor] = {'data':self.readers[sensor].GetData(index),
                                'timestamp':self.readers[sensor].GetTimestamp(index),
                                'timeofissue':self.readers[sensor].GetTimeOfIssue(index),
                                'index':index}
                                      
            self.n+=1
            return data
        else:
            raise StopIteration()
            
    def GetSensorData(self,n):
        data={}
        for sensor in self.table[n]:
            index = self.table[n][sensor]
            data[sensor] = {'data':self.readers[sensor].GetData(index),
                            'timestamp':self.readers[sensor].GetTimestamp(index),
                            'timeofissue':self.readers[sensor].GetTimeOfIssue(index),
                            'sample_number':self.readers[sensor].GetSampleNumber(index),
                            'index':index,
                            'offset_byte':self.readers[sensor].GetOffet(index)}
                                
        return data  

    def GetMostRecentOdometry(self,decoder,time):   
        
        # Steering: ID=485
        IDX = np.where(self.can_frames['ID']==485)[0]
        timediff = np.abs(self.can_frames['timestamp'][IDX] - time)
        id_steer = IDX[np.argmin(timediff)]
        message = decoder.decode([{'timestamp':self.can_frames['timestamp'][id_steer],'ID':self.can_frames['ID'][id_steer],'DATA':self.can_frames['data'][id_steer]}])
        SteeringWheel = message[0]['signals']['Steering_Wheel_Angle_deg']
        
        # Yaw: 489
        IDX = np.where(self.can_frames['ID']==489)[0]
        timediff = np.abs(self.can_frames['timestamp'][IDX] - time)
        id_yaw = IDX[np.argmin(timediff)]
        message = decoder.decode([{'timestamp':self.can_frames['timestamp'][id_yaw],'ID':self.can_frames['ID'][id_yaw],'DATA':self.can_frames['data'][id_yaw]}])
        YawRate = message[0]['signals']['YawRate_deg']
        
        # Speed: ID=1001
        IDX = np.where(self.can_frames['ID']==1001)[0]
        timediff = np.abs(self.can_frames['timestamp'][IDX] - time)
        id_speed = IDX[np.argmin(timediff)]
        message = decoder.decode([{'timestamp':self.can_frames['timestamp'][id_speed],'ID':self.can_frames['ID'][id_speed],'DATA':self.can_frames['data'][id_speed]}])
        VehSpd = message[0]['signals']['Speed_kph']

        return SteeringWheel,YawRate,VehSpd
        

    def print_info(self):
        
        print('Available sensors:')
        for sensor in self.dicts:
            if(len(self.dicts[sensor]['timestamp'])>0):
                print('# Sensor: ',sensor)
                print('\t- filename: ',self.dicts[sensor]['filename'])                  
                print('\t- Nb Samples: ',self.dicts[sensor]['sample'][-1])
                duration = (self.dicts[sensor]['timestamp'][-1]-self.dicts[sensor]['timestamp'][0])/1000000
                print('\t- Duration: ',int(duration),'sec')
                print('\t- Update rate: ',duration/self.dicts[sensor]['sample'][-1]*1000,'ms')    
            else:
                print('# Sensor: ',sensor)
                print('\t- NO DATA AVAILABLE')         
        
                

class ASyncReader():
    def __init__(self,folder):
        self.recorder_folder_path = Path(folder)

        
        rec_file_name = self.recorder_folder_path.name+'_events_log.rec'
        rec_file_path = self.recorder_folder_path/rec_file_name

        labls = np.array([str(i) for i in range(22)]) # create some row names
        df = pd.read_csv(rec_file_path,header=None,names=labls,sep='[-|\s+]',engine='python')
        nbColumn = df.shape[1]
        self.df = df.iloc[:, np.arange(1,nbColumn,4)]
        self.df.columns = ['timestamp','timeofissue', 'data_sample', 'sensor', 'offset','datasize']
        self.sensorsFilters = self.df['sensor'].unique()
        self.filters = []
        self.df_filtered = self.df
        
        print('-------------------------------------------------------------------------')
        print('- Sensors available:')
        for s in self.sensorsFilters:
            print('-    ',s)
        print('-')
        print('- You might use function "setSensorFilters" to select sensors you want to read!')
        print('-------------------------------------------------------------------------')
        
        self.dicts = parse_recording(folder)
        self.readers={}
        for sensor in self.sensorsFilters:  

            # Make sure the sensor file exist!
            if(sensor not in self.dicts.keys()):
                continue
                     
            if(sensor=='camera'):
                self.readers[sensor] = CameraReader(self.dicts[sensor])
            if(sensor=='can'):
                self.readers[sensor] = CANReader(self.dicts[sensor])
            if(sensor=='radar_ch1'):
                self.readers[sensor] = RadarReader(self.dicts[sensor])
            if(sensor=='radar_ch2'):
                self.readers[sensor] = RadarReader(self.dicts[sensor])
            if(sensor=='radar_ch3'):
                self.readers[sensor] = RadarReader(self.dicts[sensor])         
            if(sensor=='radar_ch0'):
                self.readers[sensor] = RadarReader(self.dicts[sensor])                 
            if(sensor=='scala'):
                self.readers[sensor] = LaserReader(self.dicts[sensor])
            if(sensor=='gps'):
                self.readers[sensor] = GPSReader(self.dicts[sensor])

        self.can_frames={'timestamp':[],'ID':[],'data':[]}
        if('can' in self.dicts):
            A = []
            for i in range(len(self.readers['can'].dict['offset'])):
                A.append(self.readers['can'].GetData(i))
            
            A=np.concatenate(A)
            
            for i in range(len(A)):
                self.can_frames['timestamp'].append(A[i]['timestamp'])
                self.can_frames['ID'].append(A[i]['ID'])
                self.can_frames['data'].append(A[i]['DATA'])
            self.can_frames['ID'] = np.asarray(self.can_frames['ID'])
            self.can_frames['timestamp'] = np.asarray(self.can_frames['timestamp'])
                
    def getSensorFilters(self):
        return self.sensorsFilters
    
    def setSensorFilters(self,filters=[]):

        if(len(filters)>0):
            isin = np.isin(filters,self.sensorsFilters)
            
            # Check that all the selected filters belong to the filters group 
            if(len(np.where(isin==False)[0])==0):
                # All filters are valid
                self.filters = filters
            else:
                print('Please check syntax, capital sensitive')
                print('You might check first the list of available sensors using "getSensorFilters"')
                id_ko = np.where(isin==False)[0]
                for id in id_ko:
                    print('Incorect filter: ',filters[id])
                    
                id_ok = np.where(isin==True)[0]
                self.filters=[]
                for id in id_ok:
                    self.filters.append(filters[id])
            
            print('Used filters:')
            print(self.filters)
            self.df_filtered = self.df[self.df.sensor.isin(self.filters)]
            

    def __len__(self):
        return len(self.df_filtered)
           
    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self.df_filtered):
            
            timestamp = self.df_filtered.timestamp.iloc[self.n]
            timeofissue = self.df_filtered.timeofissue.iloc[self.n]
            sample = self.df_filtered.data_sample.iloc[self.n]
            sensor = self.df_filtered.sensor.iloc[self.n]

            data = self.readers[sensor].GetData(self.n)
            
            self.n+=1
            return timestamp,timeofissue,sample,sensor,data
        else:
            raise StopIteration()

    def GetSensorData(self,index):
        timestamp = self.df_filtered.timestamp.iloc[index]
        timeofissue = self.df_filtered.timeofissue.iloc[index]
        sample = self.df_filtered.data_sample.iloc[index]
        sensor = self.df_filtered.sensor.iloc[index]

        data = self.readers[sensor].GetData(index)
        sample_number = self.readers[sensor].GetSampleNumber(index)

        return timestamp,timeofissue,sample,sensor,sample_number,data   

            
    def print_info(self):
        
        print('Available sensors:')
        for sensor in self.dicts:
            if(len(self.dicts[sensor]['timestamp'])>0):
                print('# Sensor: ',sensor)
                print('\t- filename: ',self.dicts[sensor]['filename'])                  
                print('\t- Nb Samples: ',self.dicts[sensor]['sample'][-1])
                duration = (self.dicts[sensor]['timestamp'][-1]-self.dicts[sensor]['timestamp'][0])/1000000
                print('\t- Duration: ',int(duration),'sec')
                print('\t- Update rate: ',duration/self.dicts[sensor]['sample'][-1]*1000,'ms') 
            else:
                print('# Sensor: ',sensor)
                print('\t-No DATA')      

    def GetMostRecentOdometry(self,decoder,time):   
        
        # Steering: ID=485
        IDX = np.where(self.can_frames['ID']==485)[0]
        timediff = np.abs(self.can_frames['timestamp'][IDX] - time)
        id_steer = IDX[np.argmin(timediff)]
        message = decoder.decode([{'timestamp':self.can_frames['timestamp'][id_steer],'ID':self.can_frames['ID'][id_steer],'DATA':self.can_frames['data'][id_steer]}])
        SteeringWheel = message[0]['signals']['Steering_Wheel_Angle_deg']
        
        # Yaw: 489
        IDX = np.where(self.can_frames['ID']==489)[0]
        timediff = np.abs(self.can_frames['timestamp'][IDX] - time)
        id_yaw = IDX[np.argmin(timediff)]
        message = decoder.decode([{'timestamp':self.can_frames['timestamp'][id_yaw],'ID':self.can_frames['ID'][id_yaw],'DATA':self.can_frames['data'][id_yaw]}])
        YawRate = message[0]['signals']['YawRate_deg']
        
        # Speed: ID=1001
        IDX = np.where(self.can_frames['ID']==1001)[0]
        timediff = np.abs(self.can_frames['timestamp'][IDX] - time)
        id_speed = IDX[np.argmin(timediff)]
        message = decoder.decode([{'timestamp':self.can_frames['timestamp'][id_speed],'ID':self.can_frames['ID'][id_speed],'DATA':self.can_frames['data'][id_speed]}])
        VehSpd = message[0]['signals']['Speed_kph']

        return SteeringWheel,YawRate,VehSpd
