# chuẩn
import tkinter as tk
import pandas as pd
import pickle
import os
# Function for splitting the LLM response into individual sentences
def split_string(input_string, key):
    split_list = input_string.split(key)
    split_list = [sentence.strip() for sentence in split_list if sentence.strip()]
    return split_list

# Load sentences from the pickle file
def load_sentences_from_pickle(file_path):
    with open(file_path, 'rb') as file:
        l1 = pickle.load(file)
    sentences = []
    for i in l1:
        split_sentences_list = split_string(i, '#')
        sentences.extend(split_sentences_list)
    tmp_sen = []
    for i in sentences:
        split_sentences_list = split_string(i, '\n')
        tmp_sen.extend(split_sentences_list)
    return tmp_sen

# Labeling application class
class LabelingApp:
    def __init__(self, master, sentences):
        self.master = master
        self.sentences = sentences
        self.current_sentence_index = 0
        self.labels = []

        self.create_widgets()

    def create_widgets(self):
        self.canvas = tk.Canvas(self.master)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar = tk.Scrollbar(self.master, orient="horizontal", command=self.canvas.xview)
        self.scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas.configure(xscrollcommand=self.scrollbar.set)

        self.label_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.label_frame, anchor='nw')

        self.label_frame.bind("<Configure>", self.on_frame_configure)

        self.token_labels = []
        self.option_menus = []

        self.next_button = tk.Button(self.master, text="Next", command=self.next_sentence)
        self.next_button.pack(pady=10)

        self.show_sentence()

    def on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def show_sentence(self):
        for widget in self.label_frame.winfo_children():
            widget.destroy()

        sentence = self.sentences[self.current_sentence_index]
        tokens = sentence.split()

        self.token_labels = []
        self.option_menus = []

        options = ["O", "B-pn", "I-pn"]

        for i, token in enumerate(tokens):
            token_label = tk.Label(self.label_frame, text=token)
            token_label.grid(row=0, column=i, padx=5, pady=5)
            self.token_labels.append(token_label)

            var = tk.StringVar(value="O")
            option_menu = tk.OptionMenu(self.label_frame, var, *options)
            option_menu.grid(row=1, column=i, padx=5, pady=5)
            self.option_menus.append(var)

    def next_sentence(self):
        labels = [var.get() for var in self.option_menus]
        self.labels.append(labels)

        self.current_sentence_index += 1
        if self.current_sentence_index < len(self.sentences):
            self.show_sentence()
        else:
            self.save_labels()
            self.master.quit()

    def save_labels(self):
        data = []
        for sentence, labels in zip(self.sentences, self.labels):
            labels_str = ','.join(labels)
            data.append({"Data": sentence, "Label": labels_str})

        df = pd.DataFrame(data)

        # Check if labeled_data0.csv exists and find a new file name if it does   
        file_index = 0
        file_path = "labeled_data70.csv"
        while os.path.exists(file_path):
            file_index += 1
            file_path = f"labeled_data6{file_index}.csv"

        df.to_csv(file_path, index=False)

def split_list(original_list):
    length = len(original_list)
    size = length // 3
    remainder = length % 3

    sublists = []
    start = 0

    for i in range(3):
        end = start + size + (1 if i < remainder else 0)
        sublists.append(original_list[start:end])
        start = end

    return sublists

def main():
    file_path = "label_folder\llm_response_product_name_list7.pkl"
    sentences = load_sentences_from_pickle(file_path)
    # sub_sen = split_list(sentences)
    # l = []
    # l.extend(sub_sen[1])
    # l.extend(sub_sen[2])
    print(len(sentences))
    root = tk.Tk()
    app = LabelingApp(root, sentences)
    root.mainloop()
    # root = tk.Tk()
    # app = LabelingApp(root, sentences)
    # root.mainloop()

if __name__ == "__main__":
    main()
# chuẩn