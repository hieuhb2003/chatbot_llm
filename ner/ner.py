import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from config import Config
import pickle
import difflib
class NER:
    def __init__(self):
        self.model = AutoModelForTokenClassification.from_pretrained("ner\model", num_labels=3)
        self.tokenizer = AutoTokenizer.from_pretrained("ner\model")
    def invert_tensor_to_label(self, sentence,predictions):
        token = self.tokenizer.tokenize(sentence)
        # print(type(token))
        # print(predictions[0].tolist())
        label_list = []
        if(len(predictions[0].tolist()) == len(sentence.split()) + 2):
            s = ''
            for idx,i in enumerate (predictions[0].tolist()):
                if i != 0:
                    s = s + token[idx-1] + ' '
                else:
                    if(predictions[0].tolist()[idx-1] !=0):
                        label_list.append(s.strip())
                        s = ''
        if(len(predictions[0].tolist()) > len(sentence.split()) + 2):
            i = 0
            tensor_list = predictions[0].tolist()
            s = ''
            while i < len(tensor_list):
                if tensor_list[i]!=0:
                    if(tensor_list[i-1] == 2  and tensor_list[i] == 1):
                        label_list.append(s.strip())
                        s = ''
                    if(token[i-1][-1] != '@'):
                        # print(f'{i-1} {token[i-1]}')
                        s = s + token[i-1] + ' '
                    else:
                        s = s + token[i-1][:-2]
                else:
                    if(tensor_list[i-1]!=0):
                        label_list.append(s.strip())
                        s = ''
                i = i + 1
        label_list = list(set(label_list))
        return label_list
    def classify(self, label_list):
        products = ['Bàn là', 'Máy sấy tóc', 'Bình nước nóng', 'Bình đun nước',
       'Bếp từ', 'Công tắc thông minh', 'Ghế massage daikiosan',
       'Lò vi sóng', 'Lò nướng', 'Máy Giặt', 'Máy Sấy',
       'Máy lọc không khí', 'Máy hút bụi', 'Máy lọc nước', 'Máy xay',
       'Nồi chiên không dầu', 'Nồi cơm điện', 'Nồi áp suất',
       'Robot hút bụi', 'Thiết bị Camera', 'Thiết bị Webcam',
       'Thiết bị Wifi', 'thiết bị gia dụng', 'Điều hòa',
       'Đèn Năng Lượng Mặt Trời']       
        with open(Config.product_dir, 'rb') as f:
            product_name = pickle.load(f)
        entity_product = []
        entity_product_name = []
        for item in label_list:
            match = difflib.get_close_matches(item, product_name, n=1, cutoff=0.3)
            match2 = difflib.get_close_matches(item, products, n=1, cutoff=0.3)
            score1 = 0
            score2 = 0
            if match:
                score1 = difflib.SequenceMatcher(None, item, match[0]).ratio()
            if match2:
                score2 = difflib.SequenceMatcher(None, item, match2[0]).ratio()
            if score1 > score2:
                entity_product_name.append(match[0])
            if score1 < score2:
                entity_product.append(match2[0])
        return {"GROUP_NAME": entity_product, "NAME": entity_product_name}
    def predict(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        label_list = self.invert_tensor_to_label(sentence, predictions)
        return self.classify(label_list)
# sentence = """Bạn x có tố chất thông minh cùng với sự cố gắng không ngừng nghỉ, bạn đã giành giải nhất môn toán. 
# Trong không khí vui mừng chiến thắng, bạn đã được
# bố mẹ thưởng cho một con robot đồ chơi và bàn ủi khô bluestone DIB-3776 1300W. Bàn ủi là thiết bị cần có ở mọi nhà.
# Bên cạnh đó, Kalite KL-1500 là một nồi chiên không dầu rất tốt, gia đình đang cân nhắc mua. 
# Và sản phẩm máy sấy tóc BHC010/10 cũng rất đáng chú ý"""
sentence = ""
model = NER()
print(model.predict(sentence))