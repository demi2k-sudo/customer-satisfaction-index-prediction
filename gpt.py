import openai
from dotenv import load_dotenv
load_dotenv()

class LLM:
    
    def __init__(self, api_base, api_version, api_key):
        self.api_base = api_base
        self.api_version = api_version
        self.api_key = api_key
        
    
    def get_csi(self, transcripts, emotions):
        openai.api_type = "azure"
        openai.api_base = self.api_base#"https://openai-demetrius.openai.azure.com/"
        openai.api_version = self.api_version#"2023-07-01-preview"
        openai.api_key = self.api_key#dotenv.get_key(key_to_get="OPENAI_API_KEY", dotenv_path = "F:\Software-Project\Sport-Highlights\LLM\.env")      
        example = "Communication: 8.5/10 Resolution: 8/10 Emotion Handling: 7/10. So, the overall Customer Satisfaction Index can be calculated as the average of these three scores, which is approximately 7.8/10."
        message_text = [{"role":"system","content":f"I will provide you with the transcripts of a customer service call. I will also provide you the tone of the voices at each timestamp.('a': Anger 'h': Happy 'n': Neutral) You have to analyse both and come up with a Customer Satisfaction Index<Transcripts of the talks>\n{transcripts}<Transcripts of the talks\>\n<Tone and emotion of the voice>\n{emotions}<\Tone and emotion of the voice>\n<Example>\n{example}<Example\>"}]
        self.completion = openai.ChatCompletion.create(
        engine="gpt4-demetrius",
        messages = message_text,
        temperature=0.9,
        max_tokens=5000,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
        )
    
    
        return self.completion
        



