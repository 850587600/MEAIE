import numpy as np
from collections import Counter
import json
import pickle
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn

# load a file and return a list of tuple containing $num integers in each line
def loadfile(fn, num=1):
    print('loading a file...' + fn)
    ret = []
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            x = []
            for i in range(num):
                x.append(int(th[i]))
            ret.append(tuple(x))
    return ret


def get_ids(fn):
    ids = []
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            ids.append(int(th[0]))
    return ids


def get_ent2id(fns):
    ent2id = {}
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                ent2id[th[1]] = int(th[0])
    return ent2id


# The most frequent attributes are selected to save space
def load_attr(fns, e, ent2id, topA=1000):
    cnt = {}
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                if th[0] not in ent2id:
                    continue
                for i in range(1, len(th)):
                    if th[i] not in cnt:
                        cnt[th[i]] = 1
                    else:
                        cnt[th[i]] += 1
    fre = [(k, cnt[k]) for k in sorted(cnt, key=cnt.get, reverse=True)]
    attr2id = {}
    for i in range(min(topA, len(fre))):
        attr2id[fre[i][0]] = i
    attr = np.zeros((e, topA), dtype=np.float32)
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                if th[0] in ent2id:
                    for i in range(1, len(th)):
                        if th[i] in attr2id:
                            attr[ent2id[th[0]]][attr2id[th[i]]] = 1.0
    return attr


def load_relation(e, KG, topR=1000):
    rel_mat = np.zeros((e, topR), dtype=np.float32)
    rels = np.array(KG)[:, 1]
    top_rels = Counter(rels).most_common(topR)
    rel_index_dict = {r: i for i, (r, cnt) in enumerate(top_rels)}
    for tri in KG:
        h = tri[0]
        r = tri[1]
        o = tri[2]
        if r in rel_index_dict:
            rel_mat[h][rel_index_dict[r]] += 1.
            rel_mat[o][rel_index_dict[r]] += 1.
    return np.array(rel_mat)


def load_json_embd(path):
    embd_dict = {}
    with open(path) as f:
        for line in f:
            example = json.loads(line.strip())
            vec = np.array([float(e) for e in example['feature'].split()])
            embd_dict[int(example['guid'])] = vec
    return embd_dict


def load_img(e_num, path):
    img_dict = pickle.load(open(path, "rb"))
    # init unknown img vector with mean and std deviation of the known's
    imgs_np = np.array(list(img_dict.values()))
    mean = np.mean(imgs_np, axis=0)
    std = np.std(imgs_np, axis=0)
    # img_embd = np.array([np.zeros_like(img_dict[0]) for i in range(e_num)]) # no image
    # img_embd = np.array([img_dict[i] if i in img_dict else np.zeros_like(img_dict[0]) for i in range(e_num)])
    img_embd = np.array(
        [img_dict[i] if i in img_dict else np.random.normal(mean, std, mean.shape[0]) for i in range(e_num)])
    print("%.2f%% entities have images" % (100 * len(img_dict) / e_num))
    return img_embd


def load_word2vec(path, dim=300):
    """
    glove or fasttext embedding
    """
    print('\n', path)
    word2vec = dict()
    err_num = 0
    err_list = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in tqdm(file.readlines(), desc="load word embedding"):
            line = line.strip('\n').split(' ')
            if len(line) != dim + 1:
                continue
            try:
                v = np.array(list(map(float, line[1:])), dtype=np.float64)
                word2vec[line[0].lower()] = v
            except:
                err_num += 1
                err_list.append(line[0])
                continue
    file.close()
    print("err list ", err_list)
    print("err num ", err_num)
    return word2vec


def load_char_bigram(path):
    """
    character bigrams of translated entity names
    """
    # load the translated entity names
    ent_names = json.load(open(path, "r"))
    # generate the bigram dictionary
    char2id = {}
    count = 0
    for _, name in ent_names:
        for word in name:
            word = word.lower()
            for idx in range(len(word) - 1):
                if word[idx:idx + 2] not in char2id:
                    char2id[word[idx:idx + 2]] = count
                    count += 1
    return ent_names, char2id


def load_word_char_features(node_size, word2vec_path, name_path):
    """
    node_size : ent num
    """
    word_vecs = load_word2vec(word2vec_path)
    ent_names, char2id = load_char_bigram(name_path)

    # generate the word-level features and char-level features

    ent_vec = np.zeros((node_size, 300))
    char_vec = np.zeros((node_size, len(char2id)))
    for i, name in ent_names:
        k = 0
        for word in name:
            word = word.lower()
            if word in word_vecs:
                ent_vec[i] += word_vecs[word]
                k += 1
            for idx in range(len(word) - 1):
                char_vec[i, char2id[word[idx:idx + 2]]] += 1
        if k:
            ent_vec[i] /= k
        else:
            ent_vec[i] = np.random.random(300) - 0.5

        if np.sum(char_vec[i]) == 0:
            char_vec[i] = np.random.random(len(char2id)) - 0.5
        ent_vec[i] = ent_vec[i] / np.linalg.norm(ent_vec[i])
        char_vec[i] = char_vec[i] / np.linalg.norm(char_vec[i])

    return ent_vec, char_vec


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedder = BertModel.from_pretrained('bert-base-cased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        # print('self.tokenizer1', self.tokenizer1)

    def forward(self, inputs1, inputs2):
        f1 = dict()
        # print(self.embedder)
        for input in inputs1.keys():
            tokens = self.tokenizer.encode_plus(input,max_length=27,padding='max_length',return_tensors='pt', add_special_tokens = True,truncation=True)
            # print('tokens', tokens)
            # token_id = self.tokenizer.convert_tokens_to_ids(tokens)
            # print('token_id', token_id)
            tokens_id_tensor = torch.tensor(np.array(tokens['input_ids']))
            outputs = self.embedder(tokens_id_tensor)
            # print('outputs', outputs)
            f1[input] = outputs[1]
        for input in inputs2.keys():
            tokens = self.tokenizer.encode_plus(input,max_length=27,padding='max_length',return_tensors='pt', add_special_tokens = True,truncation=True)
            # print('tokens', tokens)
            # token_id = self.tokenizer.convert_tokens_to_ids(tokens)
            # print('token_id', token_id)
            tokens_id_tensor = torch.tensor(np.array(tokens['input_ids']))
            # tokens_id_tensor= tokenizers.
            outputs = self.embedder(tokens_id_tensor)
            # print('outputs', outputs)
            f1[input] = outputs[1]
        return f1


def generate_attr_id(att_f1, att_f2):
    id_attr_dict1 = dict()
    attr_id_dict1 = dict()
    id_attr_dict2 = dict()
    attr_id_dict2 = dict()
    triples1 = set()
    triples2 = set()
    attr_embed = []
    cnt = 1
    file = open(att_f1, 'r', encoding='utf-8')
    for line in file.readlines():
        params = line.strip().split('\t')
        # print('params', params)
        assert len(params) == 3
        e = params[0].strip()
        a = params[1].strip()
        v = params[2].strip()
        triples1.add((e, a, v))
        if params[1] not in attr_id_dict1.keys():
            id_attr_dict1[cnt] = params[1]
            attr_id_dict1[params[1]] = cnt
            cnt += 1
    file.close()
    assert len(id_attr_dict1) == len(attr_id_dict1)

    file = open(att_f2, 'r', encoding='utf-8')
    for line in file.readlines():
        params = line.strip().split('\t')
        # print('params2', params)
        # assert len(params) == 3
        e = params[0].strip()
        a = params[1].strip()
        v = params[2].strip()
        if '"^^' in v:
            v = v[:v.index('"^^')]
        if '-' in v and v.count('-')>1:
            v = v.replace('-', '')
        v = v.replace('"', '')
        triples2.add((e, a, v))
        if params[1] not in attr_id_dict2.keys():
            id_attr_dict2[cnt] = params[1]
            attr_id_dict2[params[1]] = cnt
            cnt += 1
    file.close()
    model = Model()
    results = model(attr_id_dict1, attr_id_dict2)
    attr_num = len(id_attr_dict1) + len(id_attr_dict2)
    attr_embed.append(np.zeros(768))
    for i in range(1, attr_num + 1):
        if i in id_attr_dict1.keys():
            emb = np.array(results[id_attr_dict1[i]].detach().numpy())
            emb = torch.tensor(emb)
            emb = torch.squeeze(emb)
            attr_embed.append(emb)
        elif i in id_attr_dict2.keys():
            emb = np.array(results[id_attr_dict2[i]].detach().numpy())
            emb = torch.tensor(emb)
            emb = torch.squeeze(emb)
            attr_embed.append(emb)
        else:
            print("error!")
            exit()
    # print('print(len(attr_embed[1]))', len(attr_embed[1]))
    print('emb.shapeemb.shapeemb.shapeemb.shape', emb.shape)
    # print('print(12321asdasda))', len(attr_embed[0]), attr_embed[0])
    # print('print(12321asdasda))', len(attr_embed[1]), attr_embed[1])
    return attr_embed, attr_id_dict1, attr_id_dict2, triples1, triples2


def uris_attribute_triple_2ids(uris, ent_ids, att_ids):
    id_uris = list()
    e_av_dict = dict()
    for u1, u2, u3 in uris:
        assert u1 in ent_ids
        e_id = ent_ids[u1]
        assert u2 in att_ids
        a_id = att_ids[u2]
        v = u3.split('\"^^')[0].strip('\"')
        if 'e-' in v:
            pass
        elif '-' in v and v[0] != '-':
            v = v.split('-')[0]
        elif v[0] == '-' and v.count('-') > 1:
            v = '-' + v.split('-')[1]
        if '#' in v:
            v = v.strip('#')
        id_uris.append((e_id, a_id, float(v)))

        av_set = e_av_dict.get(e_id, set())
        av_set.add((a_id, float(v)))
        e_av_dict[e_id] = av_set

    assert len(id_uris) == len(set(uris))
    return id_uris, e_av_dict


def add_sup_attribute_triples(sup_links, e_av1, e_av2):
    add_attr_num1 = 0
    add_attr_num2 = 0
    for e1, e2 in sup_links:
        sup_e1 = e_av2.get(e2, set())
        sup_e2 = e_av1.get(e1, set())
        new_attr_set = sup_e1 | sup_e2
        e_av1[e1] = new_attr_set
        e_av2[e2] = new_attr_set
        add_attr_num1 += len(sup_e2)
        add_attr_num2 += len(sup_e1)
    print("sup attribute triples: {}, {}".format(add_attr_num1, add_attr_num2))
    return e_av1, e_av2


