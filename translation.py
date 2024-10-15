@router.post("/m2m_translate", response_model=TranslationResponse, tags=["Text_Translator"])
async def translate_text_m2m( request: str = Form(...), language:TargetLanguages= Form(...)):
    try:
        m2m_translator.set_domain_words(["Product Name", "Website link", "Company Name", "Target Audience"])
        m2m_translator.set_placeholders(["[Product Name]", "[Target audience]", "[Website link]", "[Discount percentage]", "[Product category]", "[Your Name]", "[Your Title]", "[Company Name]"])
        translation = m2m_translator.translate(request, language.value)
        print(translation)
        return TranslationResponse(translation=translation)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from pydantic import BaseModel

class TranslationRequest(BaseModel):
    text: str
    target_language: str

class TranslationResponse(BaseModel):
    translation: str



class M2M100Translator:
    def __init__(self, model_name: str = "facebook/m2m100_418M"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.lang_code_map = {
            'en': 'en', 'fr': 'fr', 'es': 'es', 'de': 'de', 'it': 'it', 'pt': 'pt', 
            'nl': 'nl', 'ru': 'ru', 'zh': 'zh', 'ja': 'ja', 'ko': 'ko', 'ar': 'ar',
            'pl':'pl'
        }
        self.domain_words = []
        self.placeholders = []

    def set_domain_words(self, words: List[str]):
        self.domain_words = sorted(words, key=len, reverse=True)

    def set_placeholders(self, placeholders: List[str]):
        self.placeholders = placeholders

    def detect_language(self, text: str) -> str:
        try:
            lang_code = detect(text)
            return self.lang_code_map.get(lang_code, 'en')
        except:
            return 'en'

    def split_text(self, text: str) -> List[dict]:
        parts = []
        last_end = 0
        pattern = r'\b(?:' + '|'.join(map(re.escape, self.domain_words + self.placeholders)) + r')\b'
        for match in re.finditer(pattern, text, re.IGNORECASE):
            if match.start() > last_end:
                parts.append({"type": "text", "content": text[last_end:match.start()]})
            parts.append({"type": "preserve", "content": match.group()})
            last_end = match.end()
        if last_end < len(text):
            parts.append({"type": "text", "content": text[last_end:]})
        return parts

    def translate(self, text: str, tgt_lang: str, src_lang: str = None) -> str:
        if src_lang is None:
            src_lang = self.detect_language(text)

        parts = self.split_text(text)

        print(text)
        print("Entered translate function")
        print(parts)
        
        for part in parts:
            print("Entered for loop")
            if part["type"] == "text" and part["content"].strip():
                self.tokenizer.src_lang = src_lang
                encoded = self.tokenizer(part["content"], return_tensors="pt").to(self.device)
                generated_tokens = self.model.generate(
                    **encoded,
                    forced_bos_token_id=self.tokenizer.get_lang_id(tgt_lang),
                    max_length=1024
                )
                part["content"] = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

        print("generated parts")

        result = ""
        for i, part in enumerate(parts):
            if i > 0 and part["type"] == "preserve" and not result.endswith(' '):
                result += ' '
            result += part["content"]
            if part["type"] == "preserve" and i < len(parts) - 1 and not part["content"].endswith(' '):
                result += ' '
        
        print("returning the result")

        return result.strip()

    # def translate_batch(self, texts: List[str], tgt_lang: str, src_lang: str = None) -> List[str]:
    #     return [self.translate(text, tgt_lang, src_lang) for text in texts]










INFO:     127.0.0.1:50881 - "GET /openapi.json HTTP/1.1" 200 OK
    I'm thrilled to announce the launch of our latest product, [Product Name]! This innovative solution is      designed to [Briefly describe the product's benefits and features].     [Include a brief overview of the product's key features or benefits]      We believe that [Product Name] will be a game-changer for [Target audience]. To learn more about how it can help you [Mention a specific benefit], please visit our website at [Website link].      We're offering a limited-time discount of [Discount percentage] on your first purchase. Don't miss out on this opportunity to experience the future of [Product category]!      Best regards,     [Your Name]     [Your Title]     [Company Name]
Entered translate function
[{'type': 'text', 'content': "    I'm thrilled to announce the launch of our latest product, ["}, {'type': 'preserve', 'content': 'Product Name'}, {'type': 'text', 'content': "]! This innovative solution is      designed to [Briefly describe the product's benefits and features].     [Include a brief overview of the product's key features or benefits]      We believe that ["}, {'type': 'preserve', 'content': 'Product Name'}, {'type': 'text', 'content': '] will be a game-changer for ['}, {'type': 'preserve', 'content': 'Target audience'}, {'type': 'text', 'content': ']. To learn more about how it can help you [Mention a specific benefit], please visit our website at ['}, {'type': 'preserve', 'content': 'Website link'}, {'type': 'text', 'content': "].      We're offering a limited-time discount of [Discount percentage] on your first purchase. Don't miss out on this opportunity to experience the future of [Product category]!      Best regards,     [Your Name]     [Your Title]     ["}, {'type': 'preserve', 'content': 'Company Name'}, {'type': 'text', 'content': ']'}]
Entered for loop
INFO:     127.0.0.1:50908 - "POST /api/v1/m2m_translate HTTP/1.1" 500 Internal Server Error
