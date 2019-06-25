import os
import numpy as np
import json
import sys
import gzip
import csv
import scipy
import scipy.io 
import logging
import random

class Provider:

  def __init__(self, dataset, root='.', audio_kind='mfcc', truncate=None):
    assert dataset == 'flickr8k'
    assert audio_kind == 'mfcc'
    self.root = root
    self.dataset_name = dataset
    self.audio_kind = audio_kind
    self.truncate = truncate
    self.audiofile = "{}/data/flickr8k/flickr8k_mfcc.npy".format(self.root)
    self.audio = np.load(self.audiofile, allow_pickle=True).item(0)
    self.img_feat =  scipy.io.loadmat(self.root + '/data/flickr8k/vgg_feats.mat')['feats'].T
    wav2spk = np.array(list(csv.reader(open(self.root + "/data/flickr8k/wav2spk.txt"), delimiter=' ')))
    wav2cap = np.array(list(csv.reader(open(self.root + "/data/flickr8k/wav2capt.txt"), delimiter=' ')))
    cap2wav = {}
    for row in wav2cap:
        cap2wav[row[1]+row[2]] = row[0]
    W2S = dict(list(wav2spk))
    self.dataset = json.load(open(self.root + "/data/flickr8k/dataset.json", 'r'))
    self.speakers = set()

    for image in self.dataset['images']:
            image['feat'] = self.img_feat[image['imgid']]
            for (i, sentence) in enumerate(image['sentences']):
                uttid = cap2wav["{}#{}".format(image['filename'], i)]
                
                if uttid not in self.audio:
                    logging.warning("No MFCC features, using dummy zeros for {}".format(uttid))
                    sentence['audio'] = np.zeros((10, 13))
                else:
                    sentence['audio'] = self.audio[uttid] if self.truncate is None else self.audio[uttid][:self.truncate,:]
                sentence['speaker'] = "flickr8k_" + W2S[uttid]
                self.speakers.add(sentence['speaker'])

  def makeSimEmbed(self, size=1024):
    from vg.scorer import stringsim
    ref = random.sample(list(self._iterSentences(split='train')), size)
    emb = []
    for image in self.dataset['images']:
      for sent in image['sentences']:
        emb.append(np.array([stringsim(sent['raw'], ref_i['raw']) for  ref_i in ref ]))
    np.save("{}/data/flickr8k/flickr8k_simembed-{}.npy".format(self.root, size), np.array(emb, dtype='float32'))

  def embedBySim(self, size=1024):
    try:
      emb = np.load("{}/data/flickr8k/flickr8k_simembed-{}.npy".format(self.root, size))
    except FileNotFoundError:
      self.makeSimEmbed(size=size)
      emb = np.load("{}/data/flickr8k/flickr8k_simembed-{}.npy".format(self.root, size))
    j = 0
    for image in self.dataset['images']:
      for sent in image['sentences']:
        sent['simembed'] = emb[j,:]
        j += 1
    
  def _iterImages(self, split):
        for image in self.dataset['images']:
            if image['split'] == split:
                yield image

  def _iterSentences(self, split):
        for image in self._iterImages(split):
            for sent in image['sentences']:
                yield sent
   
  def iterImages(self, split='train', shuffle=False):
        if not shuffle:
            return self._iterImages(split)
        else:
            return sorted(self._iterImages(split), key=lambda _: np.random.random())

  def iterSentences(self, split='train', shuffle=False):       
        if not shuffle:
            return self._iterSentences(split)
        else:
            return sorted(self._iterSentences(split), key=lambda _: np.random.random())

def touttid(key):
    a, b, c = key.split('_')
    return "{}_{}_{}.wav".format(a, b, c)

def getDataProvider(*args, **kwargs):
	return Provider(*args, **kwargs)

