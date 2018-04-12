__author__ = 'justinarmstrong'

import os
import pygame as pg
import keyboard
import numpy
from PIL import Image
import cv2
import random

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D

from collections import deque

import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer


ACTIONS = 6 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 10000. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 5000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 10
LEARNING_RATE = 1e-4


keybinding = {
    'action':pg.K_s,
    'jump':pg.K_a,
    'left':pg.K_LEFT,
    'right':pg.K_RIGHT,
    'down':pg.K_DOWN
}



class Control(object):


    """Control class for entire project. Contains the game loop, and contains
    the event_loop which passes events to States as needed. Logic for flipping
    states is also found here."""
    def __init__(self, caption):
        self.screen = pg.display.get_surface()
        self.done = False
        self.clock = pg.time.Clock()
        self.caption = caption
        self.fps = 60
        self.show_fps = False
        self.current_time = 0.0
        self.keys = pg.key.get_pressed()
        self.state_dict = {}
        self.state_name = None
        self.state = None
        self.choice = [True,False]
        #self.directions = ['left','right','up','down']
        self.directions = ['left','right']
        self.key_configurations = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def setup_states(self, state_dict, start_state):
        self.state_dict = state_dict
        self.state_name = start_state
        self.state = self.state_dict[self.state_name]

    def update(self):
        self.current_time = pg.time.get_ticks()
        if self.state.quit:
            self.done = True
        elif self.state.done:
            self.flip_state()
        self.state.update(self.screen, self.keys, self.current_time)

    def flip_state(self):
        previous, self.state_name = self.state_name, self.state.next
        persist = self.state.cleanup()
        self.state = self.state_dict[self.state_name]
        self.state.startup(self.current_time, persist)
        self.state.previous = previous


    def event_loop(self):

        for event in pg.event.get():

            if event.type == pg.QUIT:
                self.done = True
            elif event.type == pg.KEYDOWN:
                self.keys = pg.key.get_pressed()
                self.toggle_show_fps(event.key)
            elif event.type == pg.KEYUP:
                self.keys = pg.key.get_pressed()

            self.state.get_event(event)


    def toggle_show_fps(self, key):
        if key == pg.K_F5:
            self.show_fps = not self.show_fps
            if not self.show_fps:
                pg.display.set_caption(self.caption)

    def main(self):
        """Main loop for entire program"""

        frame_id=0;

        frame = 1;

        stack = 0;

        D = deque()

        model = createCNN()

        t = 0

        epsilon = INITIAL_EPSILON

        #state initialization
        str_image = pg.image.tostring(self.screen, "RGB",False)
        image = Image.frombytes("RGB",(self.screen.get_width(),self.screen.get_height()),str_image)
        #downsampled_image = image.resize((200,200), Image.ANTIALIAS)
        #downsampled_image.save("frames/frame_"+str(frame_id)+".png")

        x_t1 = skimage.color.rgb2gray(numpy.array(image))
        x_t1 = skimage.transform.resize(x_t1,(200,200))
        downsampled_image = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))
        downsampled_image = downsampled_image / 255.0
        downsampled_image = numpy.nan_to_num(downsampled_image) #to remove NaN values from the image

        stacked_images = numpy.stack((downsampled_image,downsampled_image,downsampled_image,downsampled_image), axis=2)
        stacked_images = stacked_images.reshape(1, stacked_images.shape[0], stacked_images.shape[1], stacked_images.shape[2]) #1x80x80x1
        s_t = stacked_images

        while not self.done:

            Q_sa = 0
            action_index = 0
            r_t = 0
            a_t = numpy.zeros([ACTIONS])

            if frame == 0:

                if random.random() <= 0.2:
                    action_index = random.randrange(ACTIONS)
                    r_t = get_reward(action_index)
                    self.keys = get_key_configurations(action_index)
                else:
                    q = model.predict(s_t)       #input a stack of 4 images, get the prediction
                    max_Q = numpy.argmax(q)
                    #print max_Q
                    action_index = max_Q
                    r_t = get_reward(action_index)
                    self.keys = get_key_configurations(action_index)

            if self.state.persist['mario dead']:
                r_t = -3.0

            if epsilon > FINAL_EPSILON and t > OBSERVE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            #Randomization Code Begins

            #direction = numpy.random.choice(self.directions)
            #s_press = numpy.random.choice(self.choice)
            #a_press = numpy.random.choice(self.choice)


            if not self.keys[pg.K_RETURN] and frame == 0:

                # if direction =='left':
                #     self.key_configurations[273] = 0
                #     self.key_configurations[274] = 0
                #     self.key_configurations[275] = 0
                #     self.key_configurations[276] = 1
                # elif direction == 'right':
                #     self.key_configurations[273] = 0
                #     self.key_configurations[274] = 0
                #     self.key_configurations[275] = 1
                #     self.key_configurations[276] = 0
                # elif direction == 'down':
                #     self.key_configurations[273] = 0
                #     self.key_configurations[274] = 1
                #     self.key_configurations[275] = 0
                #     self.key_configurations[276] = 0
                # elif direction == 'up':
                #     self.key_configurations[273] = 1
                #     self.key_configurations[274] = 0
                #     self.key_configurations[275] = 0
                #     self.key_configurations[276] = 0
                #
                # if s_press:
                #     self.key_configurations[300] = 1
                # else:
                #     self.key_configurations[300] = 0
                #
                # if a_press:
                #     self.key_configurations[97] = 1
                # else:
                #     self.key_configurations[97] = 0

                #Frame Preprocessing Starts
                str_image = pg.image.tostring(self.screen, "RGB",False)
                image = Image.frombytes("RGB",(self.screen.get_width(),self.screen.get_height()),str_image)
                #downsampled_image = image.resize((200,200), Image.ANTIALIAS)
                #downsampled_image.save("frames/frame_"+str(frame_id)+".png")

                x_t1 = skimage.color.rgb2gray(numpy.array(image))
                x_t1 = skimage.transform.resize(x_t1,(200,200))
                downsampled_image = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))
                downsampled_image = downsampled_image / 255.0
                downsampled_image = numpy.nan_to_num(downsampled_image) #to remove NaN values from the image

                if(stack==0):
                    image1 = numpy.array(downsampled_image)
                elif(stack==1):
                    image2 = numpy.array(downsampled_image)
                elif(stack==2):
                    image3 = numpy.array(downsampled_image)
                elif(stack==3):
                    image4 = numpy.array(downsampled_image)
                else:
                    stack = 0

                    stacked_images = numpy.stack((image1,image2,image3,image4), axis=2)
                    stacked_images = stacked_images.reshape(1, stacked_images.shape[0], stacked_images.shape[1], stacked_images.shape[2]) #1x80x80x1

                    D.append((s_t, action_index, r_t, stacked_images))

                    if len(D) > REPLAY_MEMORY:
                        D.popleft()

                    s_t = stacked_images

                    #print model.predict(stacked_images)

                    #model.fit(stacked_images,[0,1])

                    #stacked_images = stacked_images.reshape(1, stacked_images.shape[0], stacked_images.shape[1], stacked_images.shape[2])

                    #print stacked_images.shape
                    #D.append((stacked_images, action_index, r_t, s_t1, terminal))

                stack+=1


                #Frame Preprocessing Ends

                #self.keys = self.key_configurations

            frame_id = frame_id+1;

            frame = frame+1

            if frame == FRAME_PER_ACTION :
                frame = 0

            #Randomization Code Ends
            print t

            if t > OBSERVE:

                print 'train'
                #sample a minibatch to train on
                minibatch = random.sample(D, BATCH)

                #Now we do the experience replay
                state_t, action_t, reward_t, state_t1 = zip(*minibatch)
                state_t = numpy.concatenate(state_t)
                state_t1 = numpy.concatenate(state_t1)
                targets = model.predict(state_t)
                Q_sa = model.predict(state_t1)
                targets[range(BATCH), action_t] = reward_t + GAMMA*numpy.max(Q_sa, axis=1)

                model.train_on_batch(state_t, targets)


            t=t+1

            # if t % 1000 == 0:
            #     model.save_weights("model.h5", overwrite=True)
            #     with open("model.json", "w") as outfile:
            #         json.dump(model.to_json(), outfile)

            self.event_loop()
            self.update()
            pg.display.update()
            self.clock.tick(self.fps)
            if self.show_fps:
                fps = self.clock.get_fps()
                with_fps = "{} - {:.2f} FPS".format(self.caption, fps)
                pg.display.set_caption(with_fps)


class _State(object):
    def __init__(self):
        self.start_time = 0.0
        self.current_time = 0.0
        self.done = False
        self.quit = False
        self.next = None
        self.previous = None
        self.persist = {}

    def get_event(self, event):
        pass

    def startup(self, current_time, persistant):
        self.persist = persistant
        self.start_time = current_time

    def cleanup(self):
        self.done = False
        return self.persist

    def update(self, surface, keys, current_time):
        pass



def load_all_gfx(directory, colorkey=(255,0,255), accept=('.png', 'jpg', 'bmp')):
    graphics = {}
    for pic in os.listdir(directory):
        name, ext = os.path.splitext(pic)
        if ext.lower() in accept:
            img = pg.image.load(os.path.join(directory, pic))
            if img.get_alpha():
                img = img.convert_alpha()
            else:
                img = img.convert()
                img.set_colorkey(colorkey)
            graphics[name]=img
    return graphics


def load_all_music(directory, accept=('.wav', '.mp3', '.ogg', '.mdi')):
    songs = {}
    for song in os.listdir(directory):
        name,ext = os.path.splitext(song)
        if ext.lower() in accept:
            songs[name] = os.path.join(directory, song)
    return songs


def load_all_fonts(directory, accept=('.ttf')):
    return load_all_music(directory, accept)


def load_all_sfx(directory, accept=('.wav','.mpe','.ogg','.mdi')):
    effects = {}
    for fx in os.listdir(directory):
        name, ext = os.path.splitext(fx)
        if ext.lower() in accept:
            effects[name] = pg.mixer.Sound(os.path.join(directory, fx))
    return effects

def createCNN():
    model = Sequential()
    model.add((Conv2D(32, (8, 8), input_shape = (200, 200, 4), strides=4, activation = 'relu')))
    model.add((Conv2D(64, (4, 4), strides=2, activation = 'relu')))
    model.add((Conv2D(64, (3, 3), strides=1, activation = 'relu')))
    model.add(Flatten())
    model.add(Dense(units = 512, activation = 'relu'))
    model.add(Dense(units = 6, activation = 'linear'))
    #model.load_weights("model.h5")
    #adam = Adam(lr=LEARNING_RATE)
    model.compile(optimizer='adam', loss='mse',metrics=['accuracy'])

    return model


def get_key_configurations(action_index):
    keys = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    if(action_index==0):
        keys[276] = 1
    elif(action_index==1):
        keys[275] = 1
    elif(action_index==2):
        keys[276] = 1
        keys[97] = 1
    elif(action_index==3):
        keys[275] = 1
        keys[97] = 1
    elif(action_index==4):
        keys[276] = 1
        keys[300] = 1
    elif(action_index==5):
        keys[275] = 1
        keys[300] = 1

    return keys

def get_reward(action_index):

    reward = 0;

    if(action_index==0):
        reward = -0.5
    elif(action_index==1):
        reward_t = 0.5
    elif(action_index==2):
        reward_t = -1.0
    elif(action_index==3):
        reward_t = 1.0
    elif(action_index==4):
        reward_t = -1.0
    elif(action_index==5):
        reward_t = 1.0

    return reward
