import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import google.generativeai as genai

genai.configure(api_key="AIzaSyBp4DD6V_Bl0agssdCdEN4oxEAELfQFNdw")


# for m in genai.list_models():
#    print(m.name, m.supported_generation_methods)
class RAGChatbot:
    def __init__(self, path="qa.json"):
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        data = json.load(open(path, "r", encoding="utf-8"))
        self.questions = [x["question"] for x in data]
        self.answers = [x["answer"] for x in data]

        embeddings = self.embed_model.encode(self.questions)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings))

        # 🔥 LLM
        self.generator = pipeline("text-generation", model="distilgpt2")

    def generate(self, query):
        try:
            q_vec = self.embed_model.encode([query])
            D, I = self.index.search(q_vec, k=1)
            context = self.answers[I[0][0]]

            prompt = f"""

                Nhiệm vụ:
                - Trả lời mọi câu hỏi liên quan đến bệnh trên cây trồng và trái cây
                - Chẩn đoán bệnh dựa trên triệu chứng người dùng cung cấp
                - Giải thích nguyên nhân gây bệnh
                - Đưa ra cách xử lý và phòng tránh
                - Tư vấn chăm sóc cây trồng

                Khả năng hỗ trợ:
                - Nhận biết bệnh trên lá, thân, rễ, hoa và trái cây
                - Tư vấn sâu bệnh, nấm bệnh, vi khuẩn và thiếu dinh dưỡng
                - Hướng dẫn chăm sóc cây
                - Đưa ra khuyến nghị nông nghiệp phù hợp

                Yêu cầu:
                - Trả lời ngắn gọn, rõ ràng, dễ hiểu
                - Không dùng thuật ngữ quá phức tạp
                - Trình bày đẹp bằng emoji và gạch đầu dòng
                - Nếu chưa đủ thông tin, hãy yêu cầu người dùng cung cấp thêm triệu chứng hoặc hình ảnh
                - Nếu độ tin cậy thấp, cần cảnh báo kết quả có thể chưa chính xác
                - Nếu cây khỏe mạnh, hãy thông báo trạng thái bình thường

                Định dạng trả lời:

                🌱 Bệnh:
                [Tên bệnh hoặc tình trạng]

                📝 Mô tả:
                [Mô tả ngắn gọn]

                🔍 Triệu chứng:
                • ...
                • ...
                • ...

                ⚠️ Nguyên nhân:
                • ...
                • ...

                💊 Cách xử lý:
                • ...
                • ...

                🛡️ Phòng tránh:
                • ...
                • ...

                📌 Khuyến nghị:
                [Lời khuyên bổ sung]
            """

            model = genai.GenerativeModel("gemini-2.5-flash")

            response = model.generate_content(prompt)

            if not response or not hasattr(response, "text"):
                return "Xin lỗi, tôi chưa trả lời được."

            return response.text

        except Exception as e:
            print("ERROR:", e)
            return "⚠️ Lỗi AI, thử lại sau."