import yaml
import shutil
from com.syml.learn.music.karaoke.wav_analyzer import WavAnalyser
import os
import pickle
from random import shuffle
import numpy as np
import librosa


BINS = 1025
TIME_SLOTS = 20
SAMPLE_JUMP = 60

class MedleyDBAnalyser:

    def __init__(self,rootDir,destDir):
        print ('init MedleyDBAnalyser...')
        self._rootDir = rootDir
        self._destDir = destDir

    def generateMasks(self):
        # This is takes hours to complete

        # remove incomplete output directories
        songs = list(f for f in os.listdir(self._destDir) if os.path.isdir(s=self._destDir + f))
        print('output dirs already present...', len(songs))
        for s in songs:
            songDir = self._destDir + s + '/'
            if os.path.isfile(path=songDir + 'masks.pkl'):
                pass
            else:
                print('removing empty/incomplete dir...', songDir)
                shutil.rmtree(songDir)

        songs = list(f for f in os.listdir(self._rootDir) if os.path.isdir(s=self._rootDir + f))
        w = WavAnalyser(rootDir=self._rootDir, destDir=self._destDir)
        print('start processing songs from src dir to generate masks...')
        for song in songs:
            if os.path.isdir(s=self._destDir + song):
                print('already processed so skipping -->', song)
                continue

            print('processing -->', song)
            w.analyseYaml(song)

    def pickSampleSongs(self):
        import os
        print(os.environ['PATH'])
        destDir = self._destDir
        sampleFile = destDir + 'samples_songs.txt'
        if os.path.isfile(sampleFile):
            os.remove(sampleFile)

        # check vocal instrument mix songs
        songs = list(f for f in os.listdir(destDir) if os.path.isdir(destDir + f))
        vocalOnly = []
        instrumentOnly = []
        mixSongs = []
        sampleCount = {}
        for s in songs:
            songDir = destDir + s + '/'
            maskFile = songDir + 'masks.pkl'
            iFile = songDir + 'stem_I_normalized.wav'
            vFile = songDir + 'stem_V_normalized.wav'

            if os.path.isfile(maskFile):
                hasStemI = os.path.isfile(iFile)
                hasStemV = os.path.isfile(vFile)

                if hasStemI and not hasStemV:
                    print('instrumental only ->> ', s)
                    instrumentOnly.append(s)
                elif hasStemV and not hasStemI:
                    print('vocal only ->> ', s)
                    vocalOnly.append(s)
                else:
                    mixSongs.append(s)
                    with open(maskFile, 'rb') as file:
                        masks = pickle.load(file=file)
                        vIdealMask = masks['vMask']
                        size = np.shape(vIdealMask)
                        nTimeSlots = size[1]
                        nTimeSlots = nTimeSlots - 60
                        nSamples = int(nTimeSlots / 60)
                        sampleCount[s]=nSamples

        print('vocal only count ->', len(vocalOnly))
        print('instrument count ->', len(instrumentOnly))
        print('mix songs count ->', len(mixSongs))

        # randomize
        shuffle(mixSongs)

        # pick 50 songs
        randomSongs = []
        for i in range(0,50):
            randomSongs.append(mixSongs[i])

        totalSamples=0

        with open(sampleFile,'w') as file:
            for s in randomSongs:
                print(s,sampleCount[s])
                totalSamples += sampleCount[s]
                file.write(s +',' + str(sampleCount[s]) +'\n')
            file.write('total samples,' + str(totalSamples))

        print('total samples...'+ str(totalSamples) + '\n')


    def getSampleBatch(self,batchSize=50):
        allSamples = self.getTrainingSamples()
        sampleCount = 0
        x = np.zeros((batchSize,BINS*TIME_SLOTS),np.float32)
        y_ = np.full((batchSize,BINS*TIME_SLOTS),0,np.float32)
        for sample in allSamples:
            index = sampleCount % batchSize
            x[index, :] = sample[0]
            y_[index, :] = sample[1]
            sampleCount += 1
            if sampleCount % batchSize == 0:
                yield x , y_
        yield x,y_


    def getTrainingSamples(self):
        sampleFile = self._destDir + 'samples_songs.txt'
        sampleGenerators = []
        with open(file=sampleFile) as f:
            lines = f.readlines()
            lines = [x.strip() for x in lines ]
            for l in lines:
                if 'total samples' in l:
                    continue
                fields = l.split(',')
                song = fields[0]
                gen = self.getSamples(song=song)
                sampleGenerators.append(gen)


        for gen in sampleGenerators:
            for sample in gen:
                yield sample


    def getSamples(self,song):
        songFile = self._destDir + song + '/orig_M_normalized.wav'
        maskFile = self._destDir + song + '/masks.pkl'
        print('...loading song..',song)
        y, sr = librosa.core.load(path=songFile, sr=None)
        stftOrig = librosa.core.stft(y)
        vIdealMask=None
        with open(maskFile, 'rb') as file:
            masks = pickle.load(file=file)
            vIdealMask = masks['vMask']

        size = np.shape(stftOrig)
        nBins = size[0]
        nTimeSlots = size[1]
        nSamples = int(nTimeSlots/SAMPLE_JUMP) -1
        linearShape = (BINS*TIME_SLOTS)
        print('...picking sample from song..',song)
        index = np.arange(0,nSamples)
        shuffle(index)
        for i in np.nditer(index):
            timeStartIdx = i*SAMPLE_JUMP
            x = stftOrig[0:BINS, timeStartIdx:timeStartIdx+TIME_SLOTS]
            _y = vIdealMask[0:BINS, timeStartIdx:timeStartIdx+TIME_SLOTS]

            x = np.reshape(a=x,newshape=linearShape,order='F')
            _y = np.reshape(a=_y, newshape=linearShape, order='F')

            fn_abs = lambda t: np.float32(np.absolute(t))
            x = np.array([fn_abs(t) for t in x])

            fn_bit = lambda t: np.float32(1 if t else 0)
            _y = np.array([fn_bit(t) for t in _y])

            yield (x, _y)






if __name__ == '__main__':
    import sys
    print(sys.version)

    rootDir = '/path/to/MedleyDB/Audio/'
    destDir = '/path/to/masks/'

    dbAnalyser =  MedleyDBAnalyser(rootDir,destDir)


    count = 0
    samples  = dbAnalyser.getSamples('CelestialShore_DieForUs')
    for s in samples:
        print(count,s[0].dtype,np.shape(s[0]),s[1].dtype,np.shape(s[1]))
        count += 1

    count = 0
    # allSamples = dbAnalyser.getTrainingSamples()
    # for s in allSamples:
    #     print(count, s[0].dtype, np.shape(s[0]), s[1].dtype, np.shape(s[1]))
    #     count += 1

    # dbAnalyser.generateMasks()
    # dbAnalyser.pickSampleSongs()
    batches = dbAnalyser.getSampleBatch()
    batchCount = 0
    for s in batches:
        print(batchCount, s[0].dtype, np.shape(s[0]), s[1].dtype, np.shape(s[1]))
        batchCount += 1


    print('done...')
