import os
import re
import tkinter as tk
from tkinter import font, Text, Button, Frame, Label
import numpy as np
import torch
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


# --- Tkinter UI Application ---

class BrailleConverterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hindi to Bharati Braille Converter")
        self.root.geometry("900x600")
        self.root.configure(bg="#f0f0f0")

        # Define fonts and colors
        self.default_font = font.Font(family="Helvetica", size=12)
        self.textarea_font = font.Font(family="Arial", size=14)
        self.button_font = font.Font(family="Helvetica", size=12, weight="bold")
        self.bg_color = "#f0f0f0"
        self.text_bg_color = "#ffffff"
        self.button_color = "#007bff"
        self.button_fg_color = "#123456"

        # Create main frame
        main_frame = Frame(self.root, bg=self.bg_color, padx=20, pady=20)
        main_frame.pack(expand=True, fill="both")

        # --- Input Area ---
        input_frame = Frame(main_frame, bg=self.bg_color)
        input_frame.pack(side="left", expand=True, fill="both", padx=(0, 10))

        input_label = Label(input_frame, text="Input Hindi Text", font=self.default_font, bg=self.bg_color)
        input_label.pack(anchor="w", pady=(0, 5))

        self.input_text = Text(input_frame, wrap="word", height=15, width=40, font=self.textarea_font,
                               bg=self.text_bg_color, relief="solid", borderwidth=1, padx=10, pady=10)
        self.input_text.pack(expand=True, fill="both")

        # --- Output Area ---
        output_frame = Frame(main_frame, bg=self.bg_color)
        output_frame.pack(side="right", expand=True, fill="both", padx=(10, 0))

        output_label = Label(output_frame, text="Output Bharati Braille", font=self.default_font, bg=self.bg_color)
        output_label.pack(anchor="w", pady=(0, 5))

        self.output_text = Text(output_frame, wrap="word", height=15, width=40, font=self.textarea_font,
                                bg=self.text_bg_color, relief="solid", borderwidth=1, padx=10, pady=10)
        self.output_text.pack(expand=True, fill="both")
        self.output_text.config(state="disabled") # Make it read-only initially

        # --- Convert Button ---
        button_frame = Frame(self.root, bg=self.bg_color, pady=10)
        button_frame.pack(fill="x")

        convert_button = Button(button_frame, text="Convert to Braille", font=self.button_font,
                                bg=self.button_color, fg=self.button_fg_color, relief="flat",
                                command=self.handle_convert, padx=15, pady=8)
        convert_button.pack()


    def handle_convert(self):
        input_string = self.input_text.get("1.0", "end-1c")

        if not input_string:
            return

        try:
            words = input_string.split(" ")
            output_string = ""
            for word in words:
                output_string += get_braille(word)+"  "
        except Exception as e:
            output_string = f"An error occurred:\n{e}"

        # Display the result in the output area
        self.output_text.config(state="normal") # Enable writing
        self.output_text.delete("1.0", "end")
        self.output_text.insert("1.0", output_string)
        self.output_text.config(state="disabled") # Disable writing


if __name__ == "__main__":
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
    model = torch.load(os.path.join(dir_path,'model','bilstm_model15.pt'),weights_only=False)
    model.load_state_dict(torch.load(os.path.join(dir_path,'model','bilstm_model_dict15.pt'),weights_only=True))
    model = model.to(device)
    max_len = 32
    model.eval()
    # ---------------------------------------------

    root = tk.Tk()
    app = BrailleConverterApp(root)
    root.mainloop()










