import os
import cupy as cp
import numpy as np
import mkl_fft
from scipy import signal
import torch
import math

class CA_CFAR():
    """
    Description:
    ------------
        Cell Averaging - Constant False Alarm Rate algorithm
        Performs an automatic detection on the input range-Doppler matrix with an adaptive thresholding.
        The threshold level is determined for each cell in the range-Doppler map with the estimation
        of the power level of its surrounding noise. The average power of the noise is estimated on a
        rectangular window, that is defined around the CUT (Cell Under Test). In order the mitigate the effect
        of the target reflection energy spreading some cells are left out from the calculation in the immediate
        vicinity of the CUT. These cells are the guard cells.
        The size of the estimation window and guard window can be set with the win_param parameter.
    Implementation notes:
    ---------------------
        Implementation based on https://github.com/petotamas/APRiL
    Parameters:
    -----------
    :param win_param: Parameters of the noise power estimation window
                      [Est. window length, Est. window width, Guard window length, Guard window width]
    :param threshold: Threshold level above the estimated average noise power
    :type win_param: python list with 4 elements
    :type threshold: float
    Return values:
    --------------
    """

    def __init__(self, win_param, threshold, rd_size):
        win_width = win_param[0]
        win_height = win_param[1]
        guard_width = win_param[2]
        guard_height = win_param[3]

        # Create window mask with guard cells
        self.mask = np.ones((2 * win_height + 1, 2 * win_width + 1), dtype=bool)
        self.mask[win_height - guard_height:win_height + 1 + guard_height, win_width - guard_width:win_width + 1 + guard_width] = 0

        # Convert threshold value
        self.threshold = 10 ** (threshold / 10)

        # Number cells within window around CUT; used for averaging operation.
        self.num_valid_cells_in_window = signal.convolve2d(np.ones(rd_size, dtype=float), self.mask, mode='same')

    def __call__(self, rd_matrix):
        """
        Description:
        ------------
            Performs the automatic detection on the input range-Doppler matrix.
        Implementation notes:
        ---------------------
        Parameters:
        -----------
        :param rd_matrix: Range-Doppler map on which the automatic detection should be performed
        :type rd_matrix: R x D complex numpy array
        Return values:
        --------------
        :return hit_matrix: Calculated hit matrix
        """
        # Convert range-Doppler map values to power
        rd_matrix = np.abs(rd_matrix) ** 2

        # Perform detection
        rd_windowed_sum = signal.convolve2d(rd_matrix, self.mask, mode='same')
        rd_avg_noise_power = rd_windowed_sum / self.num_valid_cells_in_window
        rd_snr = rd_matrix / rd_avg_noise_power
        hit_matrix = rd_snr > self.threshold

        return hit_matrix
    

class RadarSignalProcessing():
    def __init__(self,path_calib_mat,method='PC',device='cpu',lib='CuPy'):
            
        # Radar parameters
        self.numSamplePerChirp = 512
        self.numRxPerChip = 4
        self.numChirps = 256
        self.numRxAnt = 16
        self.numTxAnt = 12
        self.numReducedDoppler = 16
        self.numChirpsPerLoop = 16

        try:
            self.AoA_mat = np.load(path_calib_mat,allow_pickle=True).item()
        except IOError:
            print("Error: File does not appear to exist. Please check the path: ",path_calib_mat)
            return

        
        assert lib in ['CuPy','PyTorch'] # Should be either 'CuPy' or 'PyTorch'
        self.lib = lib
        
        assert method in ['PC','RA','RD'] # Should be either 'PC' or 'RA'
        self.method = method

        if(self.method == 'PC'):
            self.CFAR_fct = CA_CFAR(win_param=(9,9,3,3), threshold=2, rd_size=(self.numSamplePerChirp,16))
            
            self.CalibMat = np.rollaxis(self.AoA_mat['Signal'],2,1).reshape(self.AoA_mat['Signal'].shape[0]*self.AoA_mat['Signal'].shape[2],self.AoA_mat['Signal'].shape[1])
        else:
            # For RA map estimation, we consider only one elevation, the one parallel to the road plan (index=5)
            self.CalibMat=self.AoA_mat['Signal'][...,5]
        
        self.device = device
        if(self.device =='cuda'):
            if(self.lib=='CuPy'):
                print('CuPy on GPU will be used to execute the processing')
                cp.cuda.Device(0).use()
                self.CalibMat = cp.array(self.CalibMat,dtype='complex64')
                self.window = cp.array(self.AoA_mat['H'][0])
            else:
                print('PyTorch on GPU will be used to execute the processing')
                self.CalibMat = torch.from_numpy(self.CalibMat).to('cuda')
                self.window = torch.from_numpy(self.AoA_mat['H'][0]).to('cuda')
            
        else:
            print('CPU will be used to execute the processing')
            self.window = self.AoA_mat['H'][0]
            
        # Build hamming window table to reduce side lobs
        hanningWindowRange = (0.54 - 0.46*np.cos(((2*math.pi*np.arange(self.numSamplePerChirp ))/(self.numSamplePerChirp -1))))
        hanningWindowDoppler = (0.54 - 0.46*np.cos(((2*math.pi*np.arange(self.numChirps ))/(self.numChirps -1))))
        self.range_fft_coef = np.expand_dims(np.repeat(np.expand_dims(hanningWindowRange,1), repeats=self.numChirps, axis=1),2)
        self.doppler_fft_coef = np.expand_dims(np.repeat(np.expand_dims(hanningWindowDoppler, 1).transpose(), repeats=self.numSamplePerChirp, axis=0),2)
    
        ## indexes shift to find Tx spots
        self.dividend_constant_arr = np.arange(0, self.numReducedDoppler*self.numChirpsPerLoop ,self.numReducedDoppler)

    
    def run(self,adc0,adc1,adc2,adc3):
        # 1. Decode the input ADC stream to buld radar complex frames
        complex_adc = self.__build_radar_frame(adc0,adc1,adc2,adc3)
    
        # 2- Remoce DC offset
        complex_adc = complex_adc - np.mean(complex_adc, axis=(0,1))

        # 3- Range FFTs
        range_fft = mkl_fft.fft(np.multiply(complex_adc,self.range_fft_coef),self.numSamplePerChirp,axis=0)
    
        # 4- Doppler FFts
        RD_spectrums = mkl_fft.fft(np.multiply(range_fft,self.doppler_fft_coef),self.numChirps,axis=1)

        if(self.method=='RD'):
            return RD_spectrums
        elif(self.method=='PC'):
            return self.__get_PCL(RD_spectrums)
        else:
            return self.__get_RA(RD_spectrums)
         
    def __build_radar_frame(self,adc0,adc1,adc2,adc3):
        frame0 = np.reshape(adc0[0::2] + 1j*adc0[1::2], (self.numSamplePerChirp,self.numRxPerChip, self.numChirps), order ='F').transpose((0,2,1))   
        frame1 = np.reshape(adc1[0::2] + 1j*adc1[1::2], (self.numSamplePerChirp,self.numRxPerChip, self.numChirps), order ='F').transpose((0,2,1))   
        frame2 = np.reshape(adc2[0::2] + 1j*adc2[1::2], (self.numSamplePerChirp,self.numRxPerChip, self.numChirps), order ='F').transpose((0,2,1))   
        frame3 = np.reshape(adc3[0::2] + 1j*adc3[1::2], (self.numSamplePerChirp,self.numRxPerChip, self.numChirps), order ='F').transpose((0,2,1))   
        return np.concatenate([frame3,frame0,frame1,frame2],axis=2)
    
    def __get_PCL(self,RD_spectrums):
        # 1- Compute power spectrum
        power_spectrum = np.sum(np.abs(RD_spectrums),axis=2)

        # 2- Apply CFAR
        # But because Tx are phase shifted of DopplerShift=16, then reduce spectrum to MaxDoppler/16 on Doppler axis
        reduced_power_spectrum = np.sum(power_spectrum.reshape(512,16,16),axis=1)
        peaks = self.CFAR_fct(reduced_power_spectrum)
        RangeBin,DopplerBin_conv = np.where(peaks>0)

        # 3- Need to find TX0 position to rebuild the MIMO spectrum in the correct order
        DopplerBin_candidates = self.__find_TX0_position(power_spectrum, RangeBin, DopplerBin_conv)
        RangeBin_candidates = [[i] for i in RangeBin]
        doppler_indexes = []
        for doppler_bin in DopplerBin_candidates:
            DopplerBinSeq = np.remainder(doppler_bin+ self.dividend_constant_arr, self.numChirps)
            DopplerBinSeq = np.concatenate([[DopplerBinSeq[0]],DopplerBinSeq[5:]]).astype('int')
            doppler_indexes.append(DopplerBinSeq)
            

        # 4- Extract and reshape the Rx * Tx matrix into the MIMO spectrum
        MIMO_Spectrum = RD_spectrums[RangeBin_candidates,doppler_indexes,:].reshape(len(DopplerBin_candidates),-1)
        MIMO_Spectrum = np.multiply(MIMO_Spectrum,self.window)
        
        # 5- AoA: maker a cross correlation between the recieved signal vs. the calibration matrix 
        # to identify azimuth and elevation angles
        ASpec=np.abs(self.CalibMat@MIMO_Spectrum.transpose())
        
        # 6- Extract maximum per (Range,Doppler) bins
        x,y = np.where(np.isnan(ASpec))
        ASpec[x,y] = 0
        az,el = np.unravel_index(np.argmax(ASpec,axis=0),(self.AoA_mat['Signal'].shape[0],self.AoA_mat['Signal'].shape[2]))
        az = np.deg2rad(self.AoA_mat['Azimuth_table'][az])
        el = np.deg2rad(self.AoA_mat['Elevation_table'][el])
        
        RangeBin = RangeBin/self.numSamplePerChirp*103.
        
        return np.vstack([RangeBin,DopplerBin_candidates,az,el]).transpose()
        

    def __get_RA(self,RD_spectrums):

        doppler_indexes = []
        for doppler_bin in range(self.numChirps):
            DopplerBinSeq = np.remainder(doppler_bin+ self.dividend_constant_arr, self.numChirps)
            DopplerBinSeq = np.concatenate([[DopplerBinSeq[0]],DopplerBinSeq[5:]])
            doppler_indexes.append(DopplerBinSeq)

        MIMO_Spectrum = RD_spectrums[:,doppler_indexes,:].reshape(RD_spectrums.shape[0]*RD_spectrums.shape[1],-1)

        if(self.device=='cpu'):
            # Multiply with Hamming window to reduce side lobes
            MIMO_Spectrum = np.multiply(MIMO_Spectrum,self.window)

            Azimuth_spec = np.abs(self.CalibMat@MIMO_Spectrum.transpose())
            Azimuth_spec = Azimuth_spec.reshape(self.AoA_mat['Signal'].shape[0],RD_spectrums.shape[0],RD_spectrums.shape[1])

            RA_map = np.sum(np.abs(Azimuth_spec),axis=2)
                    
            return RA_map.transpose()

        else:   
            
            if(self.lib=='CuPy'):
                MIMO_Spectrum = cp.array(MIMO_Spectrum)
                # Multiply with Hamming window to reduce side lobes
                MIMO_Spectrum = cp.multiply(MIMO_Spectrum,self.window).transpose()
                Azimuth_spec = cp.abs(cp.dot(self.CalibMat,MIMO_Spectrum))
                Azimuth_spec = Azimuth_spec.reshape(self.AoA_mat['Signal'].shape[0],RD_spectrums.shape[0],RD_spectrums.shape[1])
                RA_map = np.sum(np.abs(Azimuth_spec),axis=2)

                return RA_map.transpose().get()
            else:
            
                MIMO_Spectrum = torch.from_numpy(MIMO_Spectrum).to('cuda')
                # Multiply with Hamming window to reduce side lobes
                MIMO_Spectrum = torch.transpose(torch.multiply(MIMO_Spectrum,self.window),1,0).cfloat()
                Azimuth_spec = torch.abs(torch.matmul(self.CalibMat,MIMO_Spectrum))
                Azimuth_spec = Azimuth_spec.reshape(self.AoA_mat['Signal'].shape[0],RD_spectrums.shape[0],RD_spectrums.shape[1])
                RA_map = torch.sum(torch.abs(Azimuth_spec),axis=2)

                return RA_map.detach().cpu().numpy().transpose()
        
                
    def __find_TX0_position(self,power_spectrum,range_bins,reduced_doppler_bins):        
        doppler_idx = np.tile(reduced_doppler_bins,(self.numReducedDoppler,1)).transpose()+np.repeat(np.expand_dims(np.arange(0,self.numChirps,self.numReducedDoppler),0),len(range_bins),axis=0)
        doppler_idx = np.concatenate([doppler_idx,doppler_idx[:,:4]],axis=1)
        range_bins = [[r] for r in range_bins]
        cumsum = np.cumsum(power_spectrum[range_bins,doppler_idx],axis=1) 
        N = 4
        mat = (cumsum[:,N:] - cumsum[:,:-N]) / N
        section_idx = np.argmin(mat,axis=1)
        doppler_bins = section_idx*self.numReducedDoppler+reduced_doppler_bins

        return doppler_bins