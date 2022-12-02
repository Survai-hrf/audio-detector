import os
import shutil
import numpy as np
import librosa
import moviepy.editor as mp
import json
from collections import Counter


# Load the model package
import torch
from . import config
from .sed_model import SEDWrapper
from .model.htsat import HTSAT_Swin_Transformer



def audio_detection(video_id, folder, save_output):
   '''converts video to audio splits based on n seconds and performs audio detection'''
   
   
   # Load model configs
   sed_model = HTSAT_Swin_Transformer(
      spec_size=config.htsat_spec_size,
      patch_size=config.htsat_patch_size,
      in_chans=1,
      num_classes=config.classes_num,
      window_size=config.htsat_window_size,
      config = config,
      depths = config.htsat_depth,
      embed_dim = config.htsat_dim,
      patch_stride=config.htsat_stride,
      num_heads=config.htsat_num_head
      )

   model = SEDWrapper(
      sed_model = sed_model, 
      config = config,
      dataset = ''
   )

   if config.resume_checkpoint is not None:
      print("Load Checkpoint from ", config.resume_checkpoint)
      ckpt = torch.load(config.resume_checkpoint, map_location="cpu")
      ckpt["state_dict"].pop("sed_model.head.weight")
      ckpt["state_dict"].pop("sed_model.head.bias")      

   
   # AudioSet class list and model checkpoint
   data = np.load('./model_artifacts/class_hier_map.npy', allow_pickle=True)
   model_path = './model_artifacts/htsat_audioset_pretrain.ckpt'

   
   class Audio_Classification:
      def __init__(self, model_path, config):
         super().__init__()

         self.device = torch.device('cpu')
         self.sed_model = HTSAT_Swin_Transformer(
            spec_size=config.htsat_spec_size,
            patch_size=config.htsat_patch_size,
            in_chans=1,
            num_classes=config.classes_num,
            window_size=config.htsat_window_size,
            config = config,
            depths = config.htsat_depth,
            embed_dim = config.htsat_dim,
            patch_stride=config.htsat_stride,
            num_heads=config.htsat_num_head
         )
         ckpt = torch.load(model_path, map_location="cpu")
         temp_ckpt = {}
         for key in ckpt["state_dict"]:
            temp_ckpt[key[10:]] = ckpt['state_dict'][key]
         self.sed_model.load_state_dict(temp_ckpt)
         self.sed_model.to(self.device)
         self.sed_model.eval()


      def predict(self, audiofile):

         if audiofile:
            waveform, sr = librosa.load(audiofile, sr=32000)

            with torch.no_grad():
               x = torch.from_numpy(waveform).float().to(self.device)
               output_dict = self.sed_model(x[None, :], None, True)
               pred = output_dict['clipwise_output']
               pred_post = pred[0].detach().cpu().numpy()
               pred_label = np.argmax(pred_post)
               pred_prob = np.max(pred_post)

               # removes 'Speech' from classes
               if pred_label == 0:
                  pred_post = np.delete(pred_post, 0)
                  pred_label = np.argmax(pred_post)
                  pred_prob = np.max(pred_post)
                  pred_label = np.argmax(np.delete(pred_post, 0))+1
                  pred_prob = np.max(np.delete(pred_post, 0))

               # removes 'Cap gun' from classes
               if pred_label == 431:
                    pred_post = np.delete(pred_post, 431)
                    pred_label = np.argmax(pred_post)
                    pred_prob = np.max(pred_post)
                    pred_label = np.argmax(np.delete(pred_post, 431))+1
                    pred_prob = np.max(np.delete(pred_post, 431))
            return pred_label, pred_prob



   # lists of classes to include
   classes = ["Emergency vehicle", "Gunshot, gunfire", "Helicopter", "Fireworks", "Explosion", 
           "Civil defense siren", "Cough", "Motorcycle", "Race car, auto racing", 
           "Crying, sobbing", "Fire alarm", 'Vehicle', "Car", "Motor vehicle (road)",
           "Skidding", "Battle cry", "Accelerating, revving, vroom", "Police car (siren)",
           "Fire engine, fire truck (siren)", "Conversation", "Artillery fire", "Alarm",
           "Machine gun", "Crowd", "Ambulance (siren)", "Firecracker", "Whip", "Toot" "Snort",
           "Cap gun", "Fusillade", "Yell"]

   filtered_classes = {
      'Gunshot/Boom': {"Gunshot, gunfire", "Fireworks", "Explosion", "Artillery fire", "Machine gun", "Firecracker", "Whip", "Cap gun", "Fusillade",},
      'Siren/Alarm': {"Fire alarm", "Emergency vehicle", "Civil defense siren", "Police car (siren)", "Fire engine, fire truck (siren)", "Alarm", "Ambulance (siren)"},
      'Coughing/Gasping': {"Snort", "Cough"},
      'Crying': {"Crying, sobbing"},
      'Helicopter Noise': {"Helicopter"},
      'Conversation': {"Conversation"},
      'Crowd Noise': {"Crowd"},
      'Yelling': {"Battle cry", "Yell"},
      'Horn': {"Toot"},
      'Vehicle Skidding': {"Skidding"},
      'Vehicle Noise': {"Motorcycle", "Race car, auto racing", "Vehicle", "Car", "Motor vehicle (road)", "Accelerating, revving, vroom"}
   }


   video_file = f'{video_id}.mp4'


   if folder == '':
      video_file = f'temp_videodata_storage/{video_file}'
   else:
      # make directory to store audio data
      video_file = folder

   # seconds audio is split by
   n = 3

   my_clip = mp.VideoFileClip(video_file)
   remainder = int(my_clip.duration % n)
   int_seg = int(my_clip.duration - remainder)
   durations = list(range(0, int_seg+n, n))
   paired_durations = [[x, y] for x, y in zip(durations, durations[1:])]
   file_name = my_clip.filename.split('/')[1].split('.')[0]
   

   # creates temp audio folder  
   TEMP_AUDIO_STORAGE = 'src/hts_audio_transformer/audio_temp' 
   os.makedirs(TEMP_AUDIO_STORAGE, exist_ok=True)

   

   # creates audio splits
   print('splitting audio')
   
   for i in paired_durations:
      clips = my_clip.audio.subclip(t_start=i[0], t_end=i[1])
      clips.write_audiofile(f'{TEMP_AUDIO_STORAGE}/{file_name}_{i[0]}_{i[1]}.wav', logger=None)

   # clips remainder of audio file
   if remainder > 0:
      remainder_clip = my_clip.audio.subclip(t_start=paired_durations[-1][1], t_end=paired_durations[-1][1]+remainder)
      remainder_clip.write_audiofile(f'{TEMP_AUDIO_STORAGE}/{file_name}_{paired_durations[-1][1]}_{paired_durations[-1][1]+remainder}.wav', logger=None)
   

   def extract_integer(filename):
      '''reformats audio clip name in dir for proper sorting'''
      return int(filename.split('.')[0].split('_')[1])
   
   audio_list = sorted(os.listdir(TEMP_AUDIO_STORAGE), key=extract_integer)


   def GetKey(val):
      '''returns key of filtered_classes'''
      for key, value in filtered_classes.items():
         if val in value:
            return key

  
   results_dict = {}
   unfiltered = []
   # performs audio detection on clips in directory
   print('performing audio detection')

   Audiocls = Audio_Classification(model_path, config)
   for clip in audio_list:
      pred_label, pred_prob = Audiocls.predict(f'{TEMP_AUDIO_STORAGE}/{clip}')
      unfiltered.append((clip, data[pred_label][0]))


   filtered = []
   # converts class detections to per second 
   for entry in unfiltered:
      slice_ts = entry[0].split('.')[0].split('_')[1:]
      for sec in range(int(slice_ts[0]), int(slice_ts[-1])):
         filtered.append((sec+1, entry[1]))


   # filters to only classes inside filtered_classes
   audioData = [(i[0], GetKey(i[1])) for i in filtered if i[1] in classes]
   # totals number of detections per class 
   totals = dict(Counter([i[1] for i in audioData]))


   # compiles dictionary
   results_dict['uniqueId'] = my_clip.filename.split('video/')[1]
   results_dict['totals'] = totals
   results_dict['audioData'] = audioData
   results_dict['audioGraph'] = []


   # exports data to json file
   if save_output == True:
      print('exporting to JSON file')
      if not os.path.exists('output_files'):
         os.mkdir('output_files') 
      with open(f'output_files/{file_name}.json', 'w+') as file:
         json.dump(results_dict, file, ensure_ascii=False)
      
       
   # removes temp audio files 
   shutil.rmtree(TEMP_AUDIO_STORAGE)
   print("audio clips removed")

   return results_dict

