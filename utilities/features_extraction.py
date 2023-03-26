import torch
from tqdm import tqdm
import numpy as np
import copy
from sklearn.preprocessing import normalize


def compute_standard(opt, model, dataloader, evaltypes, device, **kwargs):
    evaltypes = copy.deepcopy(evaltypes)

    n_classes = opt.n_classes
    image_paths     = np.array([x[0] for x in dataloader.dataset.image_list])
    _ = model.eval()

    ###
    feature_colls  = {key:[] for key in evaltypes}

    ###
    with torch.no_grad():
        target_labels = []
        final_iter = tqdm(dataloader, desc='Embedding Data...'.format(len(evaltypes)))
        image_paths= [x[0] for x in dataloader.dataset.image_list]
        for idx,inp in enumerate(final_iter):
            input_img,target = inp[1], inp[0]
            target_labels.extend(target.numpy().tolist())
            out = model(input_img.to(device))
            #if isinstance(out, tuple): out, aux_f = out
            if isinstance(out, tuple): out, aux_f = out # out = embedding output

            ### Include embeddings of all output features
            for evaltype in evaltypes:
                if isinstance(out, dict):
                    feature_colls[evaltype].extend(out[evaltype].cpu().detach().numpy().tolist())
                else:
                    feature_colls[evaltype].extend(out.cpu().detach().numpy().tolist())


        target_labels = np.hstack(target_labels).reshape(-1,1)

    evaltype = evaltypes[0]
    features        = np.vstack(feature_colls[evaltype]).astype('float32')
    features_cosine = normalize(features, axis=1)

    return features_cosine

