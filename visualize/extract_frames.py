### Courtesy of https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames

import os
import cv2
from glob import glob
from tqdm import tqdm
from natsort import natsorted


def video_to_frames_new(video, out_video_frames):
    try:
        os.mkdir(out_video_frames)
    except:
        pass
    # use opencv to do the job
    vidcap = cv2.VideoCapture(video)
    count = 0
    while True:
        success,image = vidcap.read()
        if not success:
            vidcap.release()
            break
        cv2.imwrite(os.path.join(out_video_frames,"{:d}.jpg".format(count+1)), image)     # save frame as JPEG file
        count += 1
    print("{} images are extacted in {}.".format(count,out_video_frames))
    
    
if __name__=="__main__":

    # path to the .mp4 file or to the directory containing the .mp4 files
    input_loc = r'./data/ubnormal/animations/normal_scene_1_scenario1_3.mp4'
    # path to the folder in which the frames extracted from the video/s will be saved
    output_loc = r'./data/ubnormal/frames/Scene1_skeleton/'
    

    try:
        os.mkdir(output_loc)
    except:
        pass

    if os.path.isdir(input_loc): 

        for video in tqdm(natsorted(glob(os.path.join(input_loc, '*.mp4')))):
            out_video_frames = os.path.join(output_loc, video.split('/')[-1].split('.')[0])
            if not os.path.exists(out_video_frames):
                video_to_frames_new(video, out_video_frames) 

    else:

        out_video_frames = os.path.join(output_loc, input_loc.split('/')[-1].split('.')[0])
        if not os.path.exists(out_video_frames):
            video_to_frames_new(input_loc, out_video_frames) 