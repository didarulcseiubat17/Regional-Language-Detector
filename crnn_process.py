import sys
import language_classifier
import crnn.utils as utils
import crnn.dataset as dataset
import torch
from torch.autograd import Variable
from PIL import Image
import crnn.models.crnn as crnn
import glob
from IPython.core import display
from geotext import GeoText

def process_img(images):
    model_path = 'crnn.pth'
    alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
    model = crnn.CRNN(32, 1, 37, 256, 1)
    print('loading pretrained model from %s' % model_path)
    model.load_state_dict(torch.load(model_path))
    converter = utils.strLabelConverter(alphabet)
    transformer = dataset.resizeNormalize((100, 32))
    for image in images:
        image = Image.fromarray(image).convert('L')
        image = transformer(image)
        image = image.view(1, *image.size())
        image = Variable(image)
        model.eval()
        preds = model(image)
        _, preds = preds.max(2)
        #preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
        sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
        print('%-20s => %-20s' % (raw_pred, sim_pred))
        places = GeoText(sim_pred)
        if(len(places.cities)>0):
            print('Location Found: ')
            print (places.cities)
        else:
            print('Location Not Found')
        language_classifier.classify(sim_pred)

    return 1
