import wave;
import scipy;
import scipy.signal;
import audioop;
import librosa;
import librosa.display;
import numpy;
import itertools;
import typing;
import timeit;
import struct;
import os;
import pickle;
import yaml;


class WavAnalyser:

    def __init__(self,rootDir,destDir):
        print('scipy version...'+scipy.__version__)
        self._rootDir=rootDir
        self._destDir=destDir
        self._songName = None
        self._stems = []
        self._mixFilePath = ''
        self._mixY=None
        self._sr=None


    def loadYaml(self,songName):
        songBaseDir = self._rootDir + songName + '/'
        yamlFile = songBaseDir + songName + '_METADATA.yaml'
        with open(yamlFile, 'r') as yamlStream:
            doc = yaml.load(stream=yamlStream)

        self._songName=songName
        self._mixFilePath = songBaseDir+doc['mix_filename']

        stemDir = doc['stem_dir']
        stemDir = songBaseDir + stemDir + '/'
        stemsMap = doc['stems']
        self._stems=[]
        for k, v in stemsMap.items():
            stemFilePath = stemDir + v['filename']
            instrument = v['instrument']
            isVocal = False
            if ('singer' in instrument) or ('vocalists' in instrument):
                print('VOCAL:');
                isVocal = True
            print(stemFilePath,instrument,isVocal)
            self._stems.append( (stemFilePath,instrument,isVocal) )



    def analyseYaml(self,songName=None):

        self._songName=songName
        if songName is not None:
            self.loadYaml(songName)

        # check if stems have atleast one V
        containsVocal = False
        containsInstrument = False
        for v in self._stems:
            if v[2]:
                containsVocal=True
            else:
                containsInstrument = True

        # if containsVocal == False or containsInstrument == False:
        #     print('no vocal or no instrumental content skipping...',self._songName)
        #     return



        # load mix file
        self._loadMixFile()
        #load stems
        self._stemHolder = {}
        for v in self._stems:
            print (v);
            self._loadStem(fileName=v[0],instr=v[1],isVocal=v[2])

        self._analyze()

    def _loadMixFile(self):
        y, sr = librosa.core.load(path=self._mixFilePath, sr=None)
        self._mixY = y
        self._sr = sr

    def _loadStem(self,fileName,instr,isVocal):
        y, sr = librosa.core.load(path=fileName,sr=None)
        print (numpy.shape(y))
        self._stemHolder[fileName] = (y,sr,isVocal,instr)

    def normalizeChannel(self,y):
        maxY = numpy.max(numpy.abs(y))
        normalizedY = numpy.divide(y, maxY)
        return normalizedY

    def _analyze(self):

        dest = self._destDir + self._songName + '/'

        if os.path.isdir(dest):
            wavFileList = [f for f in os.listdir(dest)]
            for f in wavFileList:
                os.remove(dest+f)
                print('removing...'+(dest+f))
        else:
            os.mkdir(dest)


        sr=None
        size=None
        # normalize each channel
        list_normalized_v = []
        list_normalized_i = []
        for k, v in self._stemHolder.items():
            y = v[0]
            sr = v[1]
            size = numpy.shape(y)
            mono_y=librosa.to_mono(y=y)
            normalized_y=self.normalizeChannel(mono_y)
            if v[2]:
                list_normalized_v.append(normalized_y)
            else:
                list_normalized_i.append(normalized_y)

        # add all Voice
        sum_V = numpy.zeros(size, dtype=float)
        if len(list_normalized_v) > 0:
            for n in list_normalized_v:
                sum_V = numpy.add(sum_V,n)
                sum_V = self.normalizeChannel(sum_V)
            path = dest + 'stem_V_normalized.wav'
            librosa.output.write_wav(path, sum_V, v[1], norm=True)
            print('normalized V...written..' + path)

        # add all Instruments
        sum_I = numpy.zeros(size, dtype=float)
        if len(list_normalized_i) > 0:
            for n in list_normalized_i:
                sum_I = numpy.add(sum_I, n)
                sum_I = self.normalizeChannel(sum_I)
            path = dest + 'stem_I_normalized.wav'
            librosa.output.write_wav(path, sum_I, v[1], norm=True)
            print('normalized I...written..' + path)

        # final mixture M
        sum_M = numpy.add(sum_V,sum_I)
        sum_M=self.normalizeChannel(sum_M)
        path = dest + 'stem_M_normalized.wav'
        librosa.output.write_wav(path, sum_M, v[1], norm=True)
        print('normalized M...written..' + path)
        print('all normalized...')

        print('V -->> ', numpy.shape(sum_V), numpy.max(sum_V), numpy.min(sum_V))
        #self.showSpectrum(sum_V,sr,instr='V')

        print('I -->> ', numpy.shape(sum_I), numpy.max(sum_I), numpy.min(sum_I))
        #self.showSpectrum(sum_I, sr, instr='I')

        print('M -->> ', numpy.shape(sum_M), numpy.max(sum_M), numpy.min(sum_M))
        #self.showSpectrum(sum_M, sr, instr='M')

        stftV = librosa.core.stft(sum_V)
        print('V stft done...')
        stftI = librosa.core.stft(sum_I)
        print('I stft done...')
        stftM = librosa.core.stft(sum_M)
        print('M stft done...')
        stftOrig = librosa.core.stft(self._mixY)
        print('orig M stft done...')

        print ('all stft done...')

        stftShape = numpy.shape(stftM)
        nBins = stftShape[0]
        nTimeSlots = stftShape[1]

        path = dest + 'orig_M_normalized.wav'
        librosa.output.write_wav(path, self._mixY, self._sr, norm=True)
        print('orig M...written..' + path)

        vIdealMask = None
        if len(list_normalized_v) == 0 and len(list_normalized_i) > 0:
            vIdealMask = numpy.full(stftShape,False)
        elif len(list_normalized_v) > 0 and len(list_normalized_i) == 0:
            vIdealMask = numpy.full(stftShape, True)
        else:
            vIdealMask = numpy.zeros(stftShape,dtype=numpy.bool)
            print('start binary mask gen -->>')
            for i in range(0, nBins):
                print('binary mask for bin #',i)
                for j in range(0,nTimeSlots):
                    ampV = numpy.absolute(stftV[i,j])
                    ampI = numpy.absolute(stftI[i, j])
                    if ampV >= ampI:
                        vIdealMask[i, j]=1
        masks = {}
        masks['vMask'] = vIdealMask
        path = dest + 'masks.pkl'
        with open(path, 'wb') as output:
            pickle.dump(masks,output,pickle.HIGHEST_PROTOCOL)

        print('masks...written..' + path)




    def extract(self,songName=None):

        if songName is not None:
            self.loadYaml(songName=songName)

        self._loadMixFile()
        print('loaded orig mix wav file...')

        dest = self._destDir + self._songName + '/'
        maskFile = dest + 'masks.pkl'
        with open(maskFile, 'rb') as file:
            masks = pickle.load(file=file)
            vIdealMask = masks['vMask']
        print('loaded masks...')

        stftOrig = librosa.core.stft(self._mixY)
        stftShape = numpy.shape(stftOrig)
        nBins = stftShape[0]
        nTimeSlots = stftShape[1]
        print('orig M stft done...shape -->>',numpy.shape(stftShape))

        vExtraction = numpy.zeros(stftShape, dtype=complex)
        iExtraction = numpy.zeros(stftShape, dtype=complex)
        for i in range(0, nBins):
            print('extracting V and I components using mask for bin #', i)
            for j in range(0, nTimeSlots):
                if vIdealMask[i, j]:
                    vExtraction[i, j] = stftOrig[i, j]
                else:
                    iExtraction[i, j] = stftOrig[i, j]


        vEx = librosa.core.istft(vExtraction)
        print('inverse stft V done...')
        iEx = librosa.core.istft(iExtraction)
        print('inverse stft V done...')

        path = dest + 'extract_V_normalized.wav'
        librosa.output.write_wav(path, vEx, self._sr, norm=True)
        print('extract V...written..' + path)

        path = dest + 'extract_I_normalized.wav'
        librosa.output.write_wav(path, iEx, self._sr, norm=True)
        print('extract I...written..' + path)


    def showSpectrum(self,y,sr,instr=None):
        # Load sound file

        # Let's make and display a mel-scaled power (energy-squared) spectrogram
        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

        # Convert to log scale (dB). We'll use the peak power as reference.
        log_S = librosa.logamplitude(S, ref_power=numpy.max)

        import matplotlib.pyplot as plt

        # Make a new figure
        plt.figure(figsize=(12, 4))


        # Display the spectrogram on a mel scale
        # sample rate and hop length parameters are used to render the time axis
        librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

        # Put a descriptive title on the plot
        plt.title('mel power spectrogram ' + str(instr))

        # draw a color bar
        plt.colorbar(format='%+02.0f dB')

        # Make the figure layout compact
        plt.tight_layout()



    def rawWavFile(self,fileName):
        # open wav file without librosa

        raw = wave.open(fileName,'rb');
        params = raw.getparams();
        print(params);
        nFrames = raw.getnframes();
        nChannels = raw.getnchannels();
        frames = raw.readframes(nFrames);
        nSampleSize = raw.getsampwidth();
        frameRate = raw.getframerate();
        print(len(frames));
        data_per_channel = [frames[offset::nChannels] for offset in range(nChannels)];



        chnlIdx = 0
        for s in data_per_channel:
            print ((chnlIdx,audioop.max(s,nSampleSize)));
            l = int(len(s)/nSampleSize)
            print('stft start transform...chnlIdx=' + str(chnlIdx));
            if nSampleSize==2:
                print('start unpacking bytes...')
                start = timeit.timeit()
                sg = struct.iter_unpack('>h',s)
                nums = map(lambda x:x[0],sg)
                end = timeit.timeit()
                print('time for unpacking =', end - start)
                sig = numpy.fromiter(nums,numpy.int16,count=-1)
                print (sig.shape)
                print ('start stft...')
                start = timeit.timeit()
                stftTransform = scipy.signal.stft(sig, fs=frameRate, window='hann', nperseg=2048, noverlap=2048 / 4)
                end = timeit.timeit()
                print('time for stft =', end - start)
                start = timeit.timeit()
                stftTransform2 = librosa.core.stft(sig)
                end = timeit.timeit()
                print ('time for librosa stft =',end-start)
                librosa.display.specshow(stftTransform2)
            chnlIdx = chnlIdx + 1;


        raw.close();