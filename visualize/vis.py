import json
import os
from typing import Any, Hashable
from glob import glob

import numpy as np
import streamlit as st
from matplotlib import pyplot as plt, animation
from matplotlib.animation import FuncAnimation
from natsort import natsorted
from extract_frames import video_to_frames_new


class WebApp:
    """
    """
    def __init__(self, dataset_name:str):
        st.title('Visualization of {}'.format(dataset_name))


    def __call__(self, *args: Any, **kwds: Any) -> Any:
        st.video(args[0])
        # st.pyplot(args[0])


class ClipVisualizer:
    """
    Visualize clips from datasets.
    """

    bones = [(0,1), (0,2), (1,2), (1,3), (2,4), (3,5), (4,6), # head
             (5,6), (5,11), (6,12), (11,12), # body 
             (5,7), (7,9), (6,8), (8,10), # arms
             (11,13), (12,14), (13,15), (14,16)] # legs


    human_colors = ['magenta'] * 7 +\
                   ['darkorange'] * 4 +\
                   ['forestgreen', 'lime'] + ['midnightblue', 'deepskyblue'] +\
                   ['midnightblue', 'deepskyblue'] + ['forestgreen', 'lime']


    cmap = plt.get_cmap('tab20')   
    
    frame_dim_dict = {'avenue':(640,360), 'ucf_crime':(320,240), 'hr_crime':(320,240), 'ubi_fight':(), 'ubnormal':(1080,720)}
    fps_dict = {'avenue':25, 'ucf_crime':30, 'hr_crime':30, 'ubi_fight':30, 'ubnormal':30}


    def __init__(self, dataset:str, web_app:bool=False):
        """
        Init the visualizer object. To get help, call ClipVisualizer.help()

        :param dataset: name of the dataset from which visualize clips.
        """

        self.clip = None
        self.dataset = dataset.lower()
        try:
            self.frame_dim = self.frame_dim_dict[self.dataset]
        except KeyError:
            print('Unknown dataset: {}'.format(dataset))
            print('Available datasets: {}'.format(', '.join(self.frame_dim_dict.keys())))
            exit(1)

        self.web_app = None

        if web_app:
            self.web_app = WebApp(self.dataset.upper())

        
    def help(cls):
        """
        Print help information.

        :return
        """

        print('Visualize clips from datasets.')
        print('Available datasets: {}'.format(', '.join(cls.frame_dim_dict.keys())))

    
    def _groupedby_person_annotation_format(self, clip_annotation:dict) -> bool:
        """
        Check if the format of the annotation is 
        {'person_idx':{'frame1':{'keypoints':[...], 'scores':[]}, 'frame2':{'keypoints':[...]}},}

        :param clip_annotation: dictionary of the clip annotation

        :return bool
        """

        return isinstance(list(clip_annotation.values())[0], dict)


    def _change_annotation_to_vis_format(self, clip_annotation:dict) -> dict:
        """
        Change the clip annotation format to {'frame_name':[{'keypoints':[...], 'scores':int, 'idx':int},...]}

        :param clip_annotation: dictionary of the clip annotation in the format 
                                {'person_idx':{'frame1':{'keypoints':[...], 'scores':[]}, 'frame2':{'keypoints':[...]}},}
        
        :return dictionary in the right format for the visualization
        """

        vis_format_clip = {}

        for person_idx in clip_annotation.keys():

            for frame_name in clip_annotation[person_idx].keys():

                if frame_name not in vis_format_clip.keys():
                    vis_format_clip[frame_name] = []

                person_in_frame = {'keypoints':clip_annotation[person_idx][frame_name]['keypoints'],
                                   'scores':clip_annotation[person_idx][frame_name]['scores'][0] 
                                            if isinstance(clip_annotation[person_idx][frame_name]['scores'], list) 
                                            else clip_annotation[person_idx][frame_name]['scores'],
                                    'idx':int(person_idx)}
                vis_format_clip[frame_name].append(person_in_frame)

        return vis_format_clip 


    def _read_clip_annotation(self, clip_path):
        """
        Read the file containing the extracted skeletons in the clip.

        :param clip_path: path to the annotation file

        :retrun Python dictionary containing the file records
        """

        with open(clip_path, 'r') as clip_file:
            clip_annotation = json.load(clip_file)

        if self._groupedby_person_annotation_format(clip_annotation):
            clip_annotation = self._change_annotation_to_vis_format(clip_annotation)

        return clip_annotation
    

    def _get_actors_in_frame(self, frame:Hashable) -> dict:
        """
        Get the actors (i.e. the skeletons) in the given frame.

        :param frame: frame key

        :return dictionary of actors (skeletons)
        """

        actors = {}
        detected_people = self.clip[frame] if frame in self.clip.keys() else []

        for item in detected_people:
            actors[item['idx']] = np.array(item['keypoints'], dtype=np.float32).reshape(-1, 3)

        return actors

    
    def _draw_person(self, person_idx:int, keypoints:np.ndarray) -> None:
        """
        Given the keypoints of the skeleton, draw the bones with the same 
        color selected by the person_idx.

        :param person_idx: person index assigned in the annotation file
        :param keypoints: ndarray of shape (17,3) [[x1, y1, c1,], [x2, y2, c2,], ...]

        :return 
        """

        color = self.cmap(person_idx - 1)

        for bone, color in zip(self.bones, self.human_colors):
            self.ax.plot(keypoints[bone, 0], keypoints[bone, 1], color=color, linewidth=5)


    def _set_image_background(self, frame:Hashable) -> None:
        """
        """
        if self.frames_dir:
            img = plt.imread(os.path.join(self.frames_dir, frame))
            self.ax.imshow(img)
        
    
    def _draw_frame(self, frame:Hashable) -> None:
        """
        Draw the people in the current frame.

        :param frame: key of the clip_annotation dictionary

        :return 
        """

        self._set_ax()
        self._set_image_background(frame)

        actors = self._get_actors_in_frame(frame)

        for actor_idx, keypoints in actors.items():
            self._draw_person(actor_idx, keypoints)
        

    def _set_ax(self) -> None:
        """
        Set the axes for the current frame.

        :return 
        """
        self.ax.clear()
        self.ax.set_xlim(0, self.frame_dim[0]) 
        self.ax.set_ylim(self.frame_dim[1], 0)


    def _init_plot(self) -> None:
        """
        Initialize the plot.

        :return 
        """

        self.fig, self.ax = plt.subplots(figsize=(15,12))
        self._set_ax()


    def _check_and_extract_frames(self):
        """
        """
        if not os.path.exists(self.frames_dir) or len(os.listdir(self.frames_dir)) < 3:
            video_path = os.path.join(self.frames_dir.replace('frames', 'videos'), os.path.basename(self.frames_dir)+'.mp4')
            print(video_path)
            video_to_frames_new(video_path, self.frames_dir) # in case of conflicts between cv2 and matplotlib on the graphical backend, comment this line and the corresponding import directive


    def visualize(self, clip_path:str|os.PathLike, file_name='{}/{}/{}/current_animation.mp4', frames_dir:str=None, extended=None) -> None:
        """
        Visualize the animation of the skeletons in the clip.

        :param clip_path

        :return 
        """

        self.frames_dir = frames_dir
        self._check_and_extract_frames()
        self.clip = self._read_clip_annotation(clip_path)
        self._init_plot()

        if extended:
            max_frame = max(map(lambda x: int(x.split('.')[0]), self.clip.keys()))
            frames = [f'{extended}'.format(i) for i in range(1,max_frame+1)]
        else: 
            frames = natsorted(self.clip.keys())
            
        animation_ = FuncAnimation(self.fig, self._draw_frame, frames=frames, interval=self.fps_dict[self.dataset])

        writervideo = animation.FFMpegWriter(self.fps_dict[self.dataset])
        cached_animation = file_name.format('./data', self.dataset, 'animations')
        animation_.save(cached_animation, writer=writervideo)

        if self.web_app:
            plt.close()
            self.web_app(cached_animation)
        else:
            plt.show()



if __name__ == '__main__':

    vis = ClipVisualizer('ubnormal', web_app=False)


    # for alphapose_result in glob('./data/ubnormal/alphapose/*/alphapose-results-forvis-tracked.json'):
    #     clip_name = alphapose_result.split('/')[4]
    #     frames_dir = os.path.join('./data/ubnormal/frames/', clip_name)
    #     vis.visualize(alphapose_result, frames_dir=frames_dir, 
    #                   file_name=os.path.join('{}/{}/{}/', clip_name+'.mp4'), extended='{:d}.jpg')

    vis.visualize('./data/ubnormal/alphapose/normal_scene_1_scenario1_3/alphapose-results-forvis-tracked.json', frames_dir='./data/ubnormal/frames/Scene1/normal_scene_1_scenario1_3', 
    file_name='{}/{}/{}/normal_scene_1_scenario1_3.mp4', extended='{:d}.jpg')