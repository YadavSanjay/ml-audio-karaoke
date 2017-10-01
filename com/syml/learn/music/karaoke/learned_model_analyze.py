import tensorflow as tf
import logging
import librosa
import numpy as np
import pickle

BINS = 1025
SAMPLE_WIDTH = 20

learned_model_ckpt = "D:/tmp/tensorflow_logs/music/karaoke_model.ckpt"
learned_model_ckpt_meta = "D:/tmp/tensorflow_logs/music/karaoke_model.ckpt.meta"
masks_dir = "D:/Tools/ml/music/karaoke/masks/"


def generate_test_samples(song):
    songFile = masks_dir + song + '/orig_M_normalized.wav'
    print('...loading test song..', song)
    y, sr = librosa.core.load(path=songFile, sr=None)
    stftOrig = librosa.core.stft(y)
    size = np.shape(stftOrig)
    nTimeSlots = size[1]
    linearShape = (BINS * SAMPLE_WIDTH)
    print('...picking sample from test song..', song)
    print('...timeslots=' + str(nTimeSlots))
    test_samples = np.zeros((nTimeSlots,BINS*SAMPLE_WIDTH),dtype=np.float32)
    for i in range(0, nTimeSlots - SAMPLE_WIDTH):
        x = stftOrig[0:BINS, i:i+SAMPLE_WIDTH]
        x = np.reshape(a=x, newshape=linearShape, order='F')
        fn_abs = lambda t: np.float32(np.absolute(t))
        x = np.array([fn_abs(t) for t in x])
        test_samples[i] = x

    return test_samples

def analyze_test_song(songname, test_song_samples):
    input_shape = np.shape(test_song_samples)
    num_samples = input_shape[0]
    print("total samples in song=" + songname + ", #=" + str(num_samples))
    with tf.Session() as sess:
        print("loading -- checkpoint model.")
        model_saver = tf.train.import_meta_graph(learned_model_ckpt_meta)
        model_saver.restore(sess, learned_model_ckpt)
        graph = tf.get_default_graph()
        w1 = graph.get_tensor_by_name('weights/w1:0')
        b1 = graph.get_tensor_by_name('biases/b1:0')
        w2 = graph.get_tensor_by_name('weights/w2:0')
        print("loaded -- checkpoint model.")

        x1 = test_song_samples
        y1 = tf.matmul(x1, w1) + b1
        y1_sig = tf.nn.sigmoid(y1)
        y2 = tf.matmul(y1_sig, w2)
        p1 = tf.nn.sigmoid(y2)
        masks = tf.round(p1)

        learned_masks = masks.eval()

        print("..completed mask generation for test, shape=" + str(learned_masks.shape))

    return learned_masks

def generate_probabilistic_mask(ml_learned_masks ,alpha=0.5):
    learned_masks_shape = ml_learned_masks.shape
    print(learned_masks_shape)
    num_samples =  learned_masks_shape[0]
    learned_masks = np.reshape(a=ml_learned_masks, newshape=(num_samples,BINS,SAMPLE_WIDTH), order='F')
    shape = np.shape(learned_masks)
    print(shape)
    nsamples = shape[0]
    nbins = shape[1]
    probabilistic_mask = np.zeros((nbins,nsamples))
    # skip first 20 entries for mean prediction
    for i in range(19,nsamples):
        for j in range(i, i-SAMPLE_WIDTH, -1):
            learned_mask = learned_masks[j]
            learned_mask_col_indx = i - j
            for k in range(0, BINS):
                probabilistic_mask[k,i] = probabilistic_mask[k,i] + learned_mask[k,learned_mask_col_indx]

    for i in range(0,nbins):
        for j in range(0,nsamples):
            mean_prediction = np.divide(probabilistic_mask[i,j],SAMPLE_WIDTH)
            if mean_prediction > alpha:
                probabilistic_mask[i, j] = 1
            else:
                probabilistic_mask[i, j] = 0

    return probabilistic_mask

def generate_mix_for_karaoke(song, mean_prob_mask):
    print(mean_prob_mask.shape)
    songFile = masks_dir + song + '/orig_M_normalized.wav'
    print('...loading song..', song)
    y, sr = librosa.core.load(path=songFile, sr=None)
    stftOrig = librosa.core.stft(y)
    size = np.shape(stftOrig)
    nBins = size[0]
    nTimeSlots = size[1]
    v_extract = np.zeros(size,dtype=complex)
    i_extract = np.zeros(size, dtype=complex)
    print('...picking sample from song..', song)
    for i in range(0, BINS):
        for j in range(0, nTimeSlots):
            x = stftOrig[i, j]
            voicemask = mean_prob_mask[i, j]
            if voicemask > 0:
                v_extract[i,j] = x
            else:
                i_extract[i,j] = x
    maxv = np.iinfo(np.int16).max
    v_extract_file = masks_dir + song + "/ml_v_extract.wav"
    i_extract_file = masks_dir + song + "/ml_i_extract.wav"
    inverse_v_extract = librosa.core.istft(v_extract)
    inverse_i_extract = librosa.core.istft(i_extract)
    librosa.output.write_wav(v_extract_file, (inverse_v_extract * maxv).astype(np.int16), sr)
    librosa.output.write_wav(i_extract_file, (inverse_i_extract * maxv).astype(np.int16), sr)

def remove_voice_from_song(song):
    songFile = masks_dir + song + '/orig_M_normalized.wav'
    maskFile = masks_dir + song + '/masks.pkl'
    print('...loading song..', song)
    y, sr = librosa.core.load(path=songFile, sr=None)
    stftOrig = librosa.core.stft(y)
    vIdealMask = None
    with open(maskFile, 'rb') as file:
        masks = pickle.load(file=file)
        vIdealMask = masks['vMask']

    size = np.shape(stftOrig)
    nBins = size[0]
    nTimeSlots = size[1]
    v_extract = np.zeros(size,dtype=complex)
    i_extract = np.zeros(size, dtype=complex)
    print('...picking sample from song..', song)

    for i in range(0, BINS):
        for j in range(0, nTimeSlots):
            x = stftOrig[i, j]
            voicemask = vIdealMask[i, j]
            if voicemask:
                v_extract[i,j] = x
            else:
                i_extract[i,j] = x
    maxv = np.iinfo(np.int16).max
    v_extract_file = masks_dir + song + "/true_v_extract.wav"
    i_extract_file = masks_dir + song + "/true_i_extract.wav"
    inverse_v_extract = librosa.core.istft(v_extract)
    inverse_i_extract = librosa.core.istft(i_extract)
    librosa.output.write_wav(v_extract_file, (inverse_v_extract * maxv).astype(np.int16), sr)
    librosa.output.write_wav(i_extract_file, (inverse_i_extract * maxv).astype(np.int16), sr)


test_song = "HezekiahJones_BorrowedHeart"
test_samples = generate_test_samples(test_song)
ml_learned_masks = analyze_test_song(test_song,test_samples)
mean_prob_masks = generate_probabilistic_mask(ml_learned_masks)
generate_mix_for_karaoke(test_song,mean_prob_masks)
remove_voice_from_song(test_song)