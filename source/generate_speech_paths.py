# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 02:36:54 2019

@author: Vo Thanh Phuong
"""

import os
import numpy as np
import constants

def get_all_file(direct, fileType = 'wav'):
    # get all files has a type which is fileType. Use recursion
    result = []
    folder = os.listdir(direct)
    for item in folder:
        new_direct = direct + '/' + item
        if os.path.isfile(new_direct):      # if it's file, check fileType
            if item.endswith('.{}'.format(fileType)):
                result.append(new_direct)
        else:   # if it's folder, call get_all_file with this folder
            result = result + get_all_file(new_direct, fileType)
    
    return result

def generate_speech_paths(wav_folder):    
    # @root contains 1251 speakers folder
    
    paths = [constants.DEV_PATHS, constants.TEST_PATHS]
    list_speakers_file = [constants.DEV_SPEAKERS, constants.TEST_SPEAKERS]
    
    for i in range(len(paths)):
         
        path = paths[i]
        
        all_speakers = np.loadtxt(list_speakers_file[i], dtype='str')
        
        all_files = []
        
        for j in range(len(all_speakers)):
            speaker_folder = wav_folder + all_speakers[j]
            files = get_all_file(speaker_folder)
            
            for k in range(len(files)):
                all_files.append([all_speakers[j], files[k].replace(wav_folder, '')])            
        
            print('Speaker {} - Completed'.format(all_speakers[j]))

        np.savetxt(path, all_files, fmt='%s', delimiter=',')        
    return


def get_list_speaker(root):
    # vox1_dev_wav and vox1_test_wav are contained by root
    
    wav_folder = root + 'vox1_test_wav/wav/'
    test_speakers = os.listdir(wav_folder)
    np.savetxt(constants.TEST_SPEAKERS, test_speakers, fmt='%s')
    
    wav_folder = root + 'vox1_dev_wav/wav/'
    dev_speakers = os.listdir(wav_folder)
    np.savetxt(constants.DEV_SPEAKERS, dev_speakers, fmt='%s')
    return


if __name__ == "__main__":
    # root = 'E:/Learning/HCMUS/2018-2019/Khoa Luan Tot Nghiep/Data/VoxCeleb1/'
    # get_list_speaker(root)
    
    # After call get_list_speaker, join two folders vox1_dev_wav and vox1_test_wav
    # ROOT_FOLDER contains 1251 folders, each folder represent only speakers
    generate_speech_paths(constants.ROOT_FOLDER)
    