from google import genai
import os

# Khởi tạo Client với API Key của bạn
client = genai.Client(api_key="AIzaSyDLqWiDtaE20vA689Nt-XJmNOO885scCBI")

# Gửi yêu cầu tạo nội dung
response = client.models.generate_content(
    model="gemini-2.0-flash", # Bạn có thể đổi sang gemini-3-pro nếu có quyền truy cập
    contents="Hãy giải thích về lỗ đen một cách ngắn gọn cho trẻ em."
)

print(response.text)