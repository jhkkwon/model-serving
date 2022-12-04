from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
from transformers import pipeline
from min_dalle import MinDalle
import torch


model_checkpoint = "Helsinki-NLP/opus-mt-ko-en"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
text2text_model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
text2image_model = MinDalle(
    models_root='./pretrained',
    dtype=torch.float32,
    device='cpu',
    is_mega=True,
    is_reusable=True
)

def translation(input):

    translationxx_to_yy = pipeline("translation_xx_to_yy", model=text2text_model, tokenizer=tokenizer)
    prompt = translationxx_to_yy(input)[0]['translation_text']
    print(prompt)
    image = text2image_model.generate_image(
        text=prompt,
        seed=-1,
        grid_size=1,
        is_seamless=False,
        temperature=1,
        top_k=256,
        supercondition_factor=32,
        is_verbose=False
    )
    print(image)
    return image

