import os
import re
import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
import louis


class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, pad_idx, dropout=0.5):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hid_dim,batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hid_dim * 2, vocab_size)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))  
        lstm_out, _ = self.lstm(embedded)           
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)                  
        return output
    
def predict(model, input_str, max_len=32):
    with torch.no_grad():
        seq = ["<SOS>"] + list(input_str) + ["<EOS>"]
        seq = seq[:max_len] + ["<PAD>"] * (max_len - len(seq))
        input_ids = torch.tensor([[char2idx[c] for c in seq]]).to(model.embedding.weight.device)
        output = model(input_ids)
        output = output.argmax(-1).squeeze(0)
        out = ''
        for idx in output:
            if idx2char[idx.item()] in  ["<PAD>", "<SOS>"]:
                continue
            elif idx2char[idx.item()] == "<EOS>":
                break
            else:
                out+=idx2char[idx.item()]
        return out

def get_braille(text):
    exclude_prev = "कज"
    exclude_next = "षञ"
    exclude_lst = ["कष","जञ"]

    target = "्"

    for idx,t in enumerate(text):
        if t==target and idx>2 and text[idx-3:idx] in ['क्ष','ज्ञ']:
            text = text[:idx-3]+text[idx]+text[idx-3:idx]+text[idx+1:]
        elif t==target and not (idx>1 and text[idx-2:idx] in exclude_lst) and (idx+1==len(text) or text[idx-1]+text[idx+1] not in exclude_lst ):
            text = text[:idx-1]+ text[idx]+text[idx-1]+text[idx+1:]
    
    
    lst = re.split(r'(\.{3})|(?<=[़इईउऊएऐओऔ().‘‘-])(?<!़(?=[इईउऊएऐओऔ]))',text)
    
    swar_lst = ['़','्','ि','ी','ु','ा','ू','े','ै','ो','ौ']
    nukta_lst = ['ड','ढ','क','ख','ग','फ','ज']
    amb_lst = ['़', 'इ', 'ई', 'उ', 'ऊ', 'ए', 'ऐ', 'ओ', 'औ' ]
    hn_lst = ['़','्']
    
    fn = []
    for w in lst[:-1]:
        if not w:
            continue
        elif w=='...':
            fn+= [w]
        elif len(w)>2 and ((w[-2] in hn_lst and ( w[-1] in swar_lst or w[-3] in nukta_lst )) or (w[-2] in swar_lst and ( w[-1] in hn_lst))):
            fn += [w[:-3],w[-3:]]
        else:
            fn += [w[:-2],w[-2:]]
    fn += lst[-1:]
    result = ""
    for word in fn:                
        if len(word)==1:
            result += louis.translate([os.path.join(dir_path,'braille_files',"bharati_braille.cti"),
                                os.path.join(dir_path,'braille_files',"braille-patterns.cti")],word)[0]
        elif any(token in amb_lst for token in word):
            word = louis.translate([os.path.join(dir_path,'braille_files',"bharati_braille.cti"),
                                os.path.join(dir_path,'braille_files',"braille-patterns.cti")],word)[0]
            if len(word)>1:
                result+= predict(model,word,max_len)
            else:
                result += word
        else:
            
            result += louis.translate([os.path.join(dir_path,'braille_files',"bharati_braille.cti"),
                                os.path.join(dir_path,'braille_files',"braille-patterns.cti")],word)[0]
        
    return result 
    
dir_path = os.path.abspath("")


braille_symbols = []
with open(os.path.join(dir_path,'braille_files','braille_patterns.txt'),'r') as file:
    for line in file.readlines():
        cols = line.split("#")
        braille_symbols.append(cols[1][1])

symbols = sorted(set(braille_symbols + ["<PAD>", "<SOS>", "<EOS>"]))
char2idx = {ch: i for i, ch in enumerate(symbols)}
idx2char = {i: ch for ch, i in char2idx.items()}
vocab_size = len(char2idx)
pad_idx = char2idx["<PAD>"]
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BiLSTMModel(vocab_size,128,256,pad_idx)
model.load_state_dict(torch.load(os.path.join(dir_path,'model','bilstm_model_dict15.pt'),weights_only=True))
model = model.to(device)
max_len = 32
model.eval()