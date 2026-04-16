# GemmaTaiga: Offline Eco-Guardian for Khovsgol
# Powered by Gemma 4 and llama.cpp

from llama_cpp import Llama

# 1. Моделийг ачааллах (Local path-аас)
# Хөвсгөлийн тайгад сүлжээгүй тул модель төхөөрөмж дээр байх ёстой
print("GemmaTaiga системийг ачааллаж байна...")
llm = Llama(
    model_path="./models/gemma-4-2b-it-q4_k_m.gguf", # 4-bit quantized модель
    n_ctx=2048, 
    n_threads=4  # Гар утас эсвэл лаптопны CPU-д зориулсан
)

def ask_gemma_taiga(question):
    # Хөвсгөлийн байгаль орчны контекстийг өгөх (System Prompt)
    system_prompt = (
        "Та бол Хөвсгөл аймгийн тайга болон далайн бүсэд аялагчдад туслах "
        "байгаль хамгаалагч AI туслах юм. Таны зорилго бол хүмүүст байгальд "
        "ээлтэй аялахад туслах, аюулгүй байдлын зөвлөгөө өгөх юм."
    )
    
    prompt = f"System: {system_prompt}\nUser: {question}\nAssistant:"
    
    response = llm(
        prompt,
        max_tokens=512,
        echo=False
    )
    return response['choices'][0]['text']

# Жишээ асуулт
user_query = "Хөвсгөлийн тайгад төөрвөл яах вэ? Мөн гал хэрхэн аюулгүй түлэх вэ?"
print(f"Асуулт: {user_query}")
print("-" * 30)
print(ask_gemma_taiga(user_query))
