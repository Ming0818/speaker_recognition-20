# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 02:36:54 2019

@author: Vo Thanh Phuong
"""

import os
import numpy as np
import shutil

def get_all_file(direct, fileType = 'wav'):
    result = []
    folder = os.listdir(direct)
    for item in folder:
        new_direct = direct + '/' + item
        if os.path.isfile(new_direct):
            if item.endswith('.{}'.format(fileType)):
                result.append(new_direct)
        else:
            result = result + get_all_file(new_direct, fileType)
    
    return result

def generate_speech_path(root):    
    # vox1_dev_wav and vox1_test_wav are contained by root
    
    part = ['dev', 'test']
    
    for i in range(len(part)):
        wav_folder = root + 'vox1_{}_wav/wav/'.format(part[i])
        txt_folder = '../data/vox1_{}_txt/'.format(part[i])
        
        # create folder for train/test
        if os.path.isdir(txt_folder):
            shutil.rmtree(txt_folder)
        os.mkdir(txt_folder)
        
        all_speakers = os.listdir(wav_folder)
        
        for j in range(len(all_speakers)):
            speaker_folder = wav_folder + all_speakers[j]
            all_files = get_all_file(speaker_folder)
            
            for k in range(len(all_files)):
                all_files[k] = '/' + all_files[k].replace(root, '')
            
            f_name = txt_folder + all_speakers[j] + '.txt'
            np.savetxt(f_name, all_files, fmt='%s')
        
            print('Part {} - Completed {} speakers'.format(part[i], j + 1))
        
    return

if __name__ == "__main__":
    root = 'E:/Learning/HCMUS/2018-2019/Khoa Luan Tot Nghiep/Data/VoxCeleb1/'
    generate_speech_path(root)