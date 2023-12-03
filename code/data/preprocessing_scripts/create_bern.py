import torch.nn as nn
root_dir = '../../../'
vocab_dir = root_dir+'datasets/data_preprocessed/xxx/vocab/'
dir = root_dir+'datasets/data_preprocessed/xxxx/'
# emb1 = nn.Embedding(7130, 100)
# nn.init.xavier_uniform_(emb1.weight)
# fp1 = open(dir + "entity2vec.bern", "w")
# for i in list(emb1.weight.data):
#     fp1.write(str(i)+"\n")
# fp1.close()
emb2 = nn.Embedding(234, 100)
nn.init.xavier_uniform_(emb2.weight)
fp2 = open(dir + "relation2vec.bern", "w")
for i in list(emb2.weight.data):
    fp2.write(str(i)+"\n")
fp2.close()

