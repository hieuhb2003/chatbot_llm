# from config import Config
# import csv
# import os
# from sentence_transformers import SentenceTransformer
# from pyvi.ViTokenizer import tokenize
# from tqdm import tqdm
# import numpy as np
# import pickle
# class Csv2Text:
#     def __init__(self):
#         self.meta_corpus = self.getMetaCorpus()
#         self.embeddings_corpus = self.getEmbeddMetaCorpus()
        
#     def split_text_into_chunks(text, chunk_size=100, window_size=50):
#         words = text.split()
#         num_words = len(words)
#         chunks = []
#         start_idx = 0
#         while True:
#             end_idx = start_idx + chunk_size
#             chunk = " ".join(words[start_idx:end_idx])
#             chunks.append(chunk)
#             if end_idx >= num_words:
#                 break
#             start_idx += window_size
#         return chunks
        
    
#     def getMetaCorpus(self):
#         if not os.path.exists(Config.meta_corpus_dir):
#             corpus = []
#             meta_corpus = []
#             _id = 0
#             docs = {}
#             with open(Config.csv_dir, mode='r',encoding='utf-8') as file:
#                 csv_reader = csv.reader(file)
#                 print(type(csv_reader))
#                 i = 0
#                 text = ''
#                 for row in csv_reader:
#                     if(i == 0): i += 1
#                     else:
#                         id = row[2]
#                         loai_sp = row[3]
#                         code = row[4]
#                         gia_chua_vat1 = row[5]
#                         gia_gom_vat1 = row[6]
#                         hoa_hong1 = row[7]
#                         don_hang1 = row[8]
#                         gia_chua_vat2 = row[9]
#                         gia_gom_vat2 = row[10]
#                         hoa_hong2 = row[11]
#                         don_hang2 = row[12]
#                         gia_chua_vat3 = row[13]
#                         gia = gia_gom_vat3 = row[14]
#                         hoa_hong3 = row[15]
#                         ten_sp = row[16]
#                         mo_ta = row[17]
#                         da_ban = row[19]
#                         mo_ta = mo_ta.split('\n')
#                         mo_ta = [s  for s in mo_ta if s!= '']
#                         mo_ta = [s[2:].strip() if s[0] == '•' else s.strip() for s in mo_ta]
#                         txt_mota = ''
#                         for s in mo_ta:
#                             txt_mota += s + '. '
#                         tmp = f'Sản phẩm {ten_sp} thuộc loại {loai_sp}. Sản phẩm có id = {id} và mã {code}. Đây là mô tả của sản phẩm {ten_sp} bao gồm: {txt_mota}Giá bán chính thức của {ten_sp} là {gia_gom_vat3}, giá chưa có VAT là {gia_chua_vat3}. Đặc biệt giá ưu đãi của {ten_sp} sẽ là {gia_gom_vat1}, giá chưa có VAT là {gia_chua_vat1} khi {don_hang1}. Bên cạnh đó {don_hang2}, giá ưu đãi của {ten_sp} sẽ là {gia_gom_vat2}, giá chưa có VAT là {gia_chua_vat2}. Hiện tại cửa hàng đã bán được {da_ban} sản phẩm {ten_sp}.'
#                         tmp = tmp.strip()
#                         chunks = self.split_text_into_chunks(tmp, chunk_size=200, window_size=200)
#                         title = ten_sp.strip()
#                         chunks = [f"Title: {title}\n\n{chunk}" for chunk in chunks]
#                         meta_chunks = [{
#                             "title": title,
#                             "passage": chunks[i],
#                             "id": _id + i,
#                             "len": len(chunks[i].split())
#                         } for i in range(len(chunks))]
#                         _id += len(chunks)
#                         corpus.extend(chunks)
#                         meta_corpus.extend(meta_chunks)
#                 with open("data\meta_corpus.pkl", 'wb') as f:
#                     pickle.dump(meta_corpus, f)
#         else:
#             with open(Config.meta_corpus_dir, 'rb') as f:
#                 meta_corpus = pickle.load(f)
#         return meta_corpus
    
#     def getEmbeddMetaCorpus(self):
#         if not os.path.exists(Config.embeddings_corpus_dir):
#             model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')
#             segmented_corpus = [tokenize(example["passage"]) for example in tqdm(self.meta_corpuss)]
#             embeddings = model.encode(segmented_corpus)
#             embeddings /= np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
#             with open('data\corpus_embedding.pkl', 'wb') as f:
#                 pickle.dump(embeddings, f)
#         else:
#             with open(Config.embeddings_corpus_dir, 'rb') as f:
#                 corpus = pickle.load(f)
#             return corpus

from config import Config
import csv
import os
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize
from tqdm import tqdm
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter
from qdrant_client.http import models
from dotenv import load_dotenv

load_dotenv()
class ProductVectorProcessor:
    def __init__(self):
        # Khởi tạo Qdrant client với cloud configuration
        self.qdrant = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        self.collection_name = "product_chunks"
        self.setup_collection()
        
        # Lấy corpus và embeddings
        self.meta_corpus = self.getMetaCorpus()
        self.embeddings_corpus = self.getEmbeddMetaCorpus()

    def setup_collection(self):
        try:
            self.qdrant.get_collection(self.collection_name)
        except:
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=768,
                    distance=models.Distance.COSINE,
                    on_disk=True
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=0,
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                )
            )

    def getMetaCorpus(self):
        """Lấy meta corpus từ Qdrant"""
        try:
            # Lấy tất cả points từ collection
            scroll_results = self.qdrant.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Điều chỉnh theo số lượng documents
                with_payload=True,
                with_vectors=False  # Không cần lấy vectors
            )
            
            # Chuyển đổi kết quả thành meta_corpus
            meta_corpus = []
            for point in scroll_results[0]:
                meta_corpus.append({
                    "title": point.payload["title"],
                    "passage": point.payload["passage"],
                    "id": point.id,
                    "len": point.payload["len"]
                })
            
            return meta_corpus
            
        except Exception as e:
            print(f"Lỗi khi lấy meta corpus: {e}")
            # Nếu chưa có dữ liệu, xử lý và upload dữ liệu mới
            self.process_data()
            # Gọi lại hàm để lấy dữ liệu
            return self.getMetaCorpus()

    def getEmbeddMetaCorpus(self):
        """Lấy embeddings từ Qdrant"""
        try:
            # Lấy tất cả points từ collection
            scroll_results = self.qdrant.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Điều chỉnh theo số lượng documents
                with_payload=False,
                with_vectors=True  # Cần lấy vectors
            )
            
            # Chuyển đổi kết quả thành numpy array
            embeddings = []
            for point in scroll_results[0]:
                embeddings.append(point.vector)
            
            return np.array(embeddings)
            
        except Exception as e:
            print(f"Lỗi khi lấy embeddings corpus: {e}")
            # Nếu chưa có dữ liệu, xử lý và upload dữ liệu mới
            self.process_data()
            # Gọi lại hàm để lấy dữ liệu
            return self.getEmbeddMetaCorpus()

    def process_data(self):
        """Xử lý và upload dữ liệu lên Qdrant"""
        model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')
        points = []
        _id = 0
        
        print("Bắt đầu xử lý dữ liệu...")
        with open(Config.csv_dir, mode='r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            
            for row in tqdm(csv_reader, desc="Xử lý các dòng CSV"):
                id, loai_sp = row[2], row[3]
                code, ten_sp = row[4], row[16]
                gia_chua_vat1, gia_gom_vat1 = row[5], row[6]
                gia_chua_vat2, gia_gom_vat2 = row[9], row[10]
                gia_chua_vat3, gia_gom_vat3 = row[13], row[14]
                don_hang1, don_hang2 = row[8], row[12]
                mo_ta, da_ban = row[17], row[19]

                mo_ta = mo_ta.split('\n')
                mo_ta = [s.strip() for s in mo_ta if s != '']
                mo_ta = [s[2:].strip() if s[0] == '•' else s for s in mo_ta]
                txt_mota = '. '.join(mo_ta)

                text = f'Sản phẩm {ten_sp} thuộc loại {loai_sp}. Sản phẩm có id = {id} và mã {code}. Đây là mô tả của sản phẩm {ten_sp} bao gồm: {txt_mota}Giá bán chính thức của {ten_sp} là {gia_gom_vat3}, giá chưa có VAT là {gia_chua_vat3}. Đặc biệt giá ưu đãi của {ten_sp} sẽ là {gia_gom_vat1}, giá chưa có VAT là {gia_chua_vat1} khi {don_hang1}. Bên cạnh đó {don_hang2}, giá ưu đãi của {ten_sp} sẽ là {gia_gom_vat2}, giá chưa có VAT là {gia_chua_vat2}. Hiện tại cửa hàng đã bán được {da_ban} sản phẩm {ten_sp}.'

                chunks = self.split_text_into_chunks(text.strip(), chunk_size=200, window_size=200)
                
                for chunk in chunks:
                    chunk_with_title = f"Title: {ten_sp}\n\n{chunk}"
                    
                    embedding = model.encode(tokenize(chunk_with_title))
                    embedding = embedding / np.linalg.norm(embedding)
                    
                    point = PointStruct(
                        id=_id,
                        vector=embedding.tolist(),
                        payload={
                            "title": ten_sp,
                            "passage": chunk_with_title,
                            "len": len(chunk_with_title.split()),
                            "product_id": id,
                            "product_code": code
                        }
                    )
                    points.append(point)
                    _id += 1

                if len(points) >= 100:
                    print(f"Đang upload batch {len(points)} points...")
                    self.qdrant.upsert(
                        collection_name=self.collection_name,
                        points=points,
                        wait=True
                    )
                    points = []

        if points:
            print(f"Đang upload batch cuối cùng {len(points)} points...")
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
        
        print("Hoàn thành quá trình upload dữ liệu!")

    def split_text_into_chunks(self, text, chunk_size=100, window_size=50):
        words = text.split()
        num_words = len(words)
        chunks = []
        start_idx = 0
        while True:
            end_idx = start_idx + chunk_size
            chunk = " ".join(words[start_idx:end_idx])
            chunks.append(chunk)
            if end_idx >= num_words:
                break
            start_idx += window_size
        return chunks
