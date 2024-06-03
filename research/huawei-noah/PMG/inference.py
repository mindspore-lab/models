import re
import numpy as np
import clip
# from model import InferenceModel
import mindspore as ms
from PMG_utils import loadLlama, loadSDPipeline, loadCLIP, initEnv
from prompts import NEGATIVE_PROMPT, STYLE_PROMPT, STYLE_PROMPT_SOFT, ITEM_PROMPT, STYLE_PROMPT_ATTRIBUTE, STYLE_PROMPT_ATTRIBUTE_INTEGRATE
# from compel import Compel

W_P = {
    'movie' : 2,
    'emoticon': 2
}
LLaMA_PATH = "/home/ma-user/work/code/LLaMA/2-7B-chat_hf/"
class PMGInference:
    def __init__(self):
        self.llama_config = initEnv()
        self.llama_tokenizer, self.llama_model = loadLlama(self.llama_config)

        self.num_image_prompt = 4
        self.num_prefix_prompt = 4
        self.cuda_id = 0
        
        self.soft_model = {}
        
        self.base_seed = 0
        self.style_keyword_num = 8
        self.sd_scene_now = None
    
    # The generation of emoticon is different from other scenes, so use function generateCharacter (as prference keywords) and generateEmoKeywords (as target keywords) here.
    def generateCharacter(self, item_list):
        prompt = STYLE_PROMPT['emoticon']
        prompt = prompt.replace('<Description/>', ' '.join(["%d. \"%s\";"%(i+1, x['name']) for i, x in enumerate(item_list)]))
        print(prompt)
        token = self.llama_tokenizer(prompt, return_tensors='pt').input_ids.cuda()
        output = self.llama_model.generate(token, max_new_tokens=100, do_sample=True, temperature=0.3)
        output_text = self.llama_tokenizer.decode(output[0][token.shape[-1]-4:]).split('###')[0]
        print(output_text)
        new_character = re.findall('\d. \"([^\"]*)\"', output_text)
        print(new_character)
        
        return new_character[:1]
    
    def generateEmoKeywords(self, conv):
        names = ['Alice', 'Bob']
        prompt = ITEM_PROMPT['emoticon'].format(*names)
        prompt = prompt.replace('<Conversation/>', ' '.join([' #'+names[(len(conv)-i-1)%2]+': '+s for i,s in enumerate(conv)]))
        # print(prompt)
        token = self.llama_tokenizer(prompt, return_tensors='pt').input_ids.cuda()
        output = self.llama_model.generate(token, max_new_tokens=50, do_sample=True, temperature=0.3)
        output_text = self.llama_tokenizer.decode(output[0][token.shape[-1]:]).split('###')[0]
        print(output_text)

        keywords = re.findall(r'Emotion: ([^;\"]*);', output_text)
        keywords = ', '.join(keywords)
        exp_keywords = re.findall(r'Expression: ([^;\.\"]*)[;\.\"]', output_text) + re.findall(r'Posture: ([^;\.\"]*)[;\.\"]', output_text)
        exp_keywords = ', '.join(exp_keywords)
        return [keywords, exp_keywords]
    
    def clipWork(self, images, text):
        self.CLIP_model, self.CLIP_processor = loadCLIP()
        image = ms.ops.Concat(0)([ms.Tensor(self.CLIP_processor(img)) for img in images])
        print(image.shape)
        text = clip.tokenize(text)

        logits_per_image, logits_per_text = self.CLIP_model(image, text)
        del self.CLIP_model, self.CLIP_processor
        return logits_per_image, logits_per_text
        
    # generation images multiple times with different weights and select the best one
    def calcBestImage(self, scene, item_text, style_text, gen_image_list):
        print(gen_image_list, item_text, style_text)
        logits_item, logits_style = self.clipWork(gen_image_list, [item_text, style_text])[1]
        print(logits_item, logits_style)
        
        # print(np.log(logits_item), np.log(logits_style), np.log(logits_item) + 2*np.log(logits_style))
        best_img_id = np.argmax(np.log(logits_item) + W_P[scene]*np.log(logits_style))
        print(best_img_id)
        return best_img_id
    
    # generate image using the input target item ketwords (item_kw) and preference keywords (style_kw)
    def generateImage(self, scene, item_kw=[], style_kw=[], soft_prompt=None, num=1, style_weight=2, inference_steps=30, seed=0):
        if self.sd_scene_now != scene:
            self.sd_scene_now = scene
            self.sd_pipeline = loadSDPipeline(scene)
        
        sd_text = ', '.join(item_kw)
        
        if style_weight > 0:
            for i in range(style_weight):
                sd_text += ', ' + ', '.join(style_kw)
        prompt = sd_text
        
        assert soft_prompt is None, "Soft prompt is not done yet"
            
        negative_prompt = NEGATIVE_PROMPT[scene]
        print(prompt, negative_prompt)
        gen_image_list = []
        for i in range(num):
            images = self.sd_pipeline.generate(prompt, negative_prompt, sampling_steps=inference_steps)
            gen_image_list.append(images[0])
        
        return gen_image_list

    # generate preference keywords (stylePrompt) and soft prompt (stylePromptSoft) by LLM
    def itemProcess(self, scene, item):
        if scene == 'costume':
            return item['name']
        elif scene == 'movie':
            return '%s(%s): %s.'%(item['name'], item['genre'], ', '.join(item['keyword'][:5]))
            
        
    def stylePrompt(self, scene, item_list, attributes = None):
        items_text = ' '.join(['%d. %s'%(i, self.itemProcess(scene, item)) for i, item in enumerate(item_list)])
        if attributes is None:
            prompt = STYLE_PROMPT[scene]
            prompt = prompt.replace('<Keyword_num/>', str(self.style_keyword_num))
            prompt = prompt.replace('<Items/>', items_text)
            # tokens = self.llama_tokenizer(prompt, return_tensors="pt",).input_ids.cuda(self.cuda_id)
            # output_token = self.llama_model.generate(tokens, max_new_tokens=100).cpu()
            # output_text = self.llama_tokenizer.decode(output_token[0][tokens.shape[1]:])
            inputs_ids = self.llama_tokenizer(prompt, max_length=self.llama_config.seq_length, padding="max_length")["input_ids"]
            output_text = self.llama_model.generate(inputs_ids,
                                                    max_new_tokens=self.llama_config.max_new_tokens,
                                                    do_sample=self.llama_config.do_sample,
                                                    top_k=self.llama_config.top_k,
                                                    top_p=self.llama_config.top_p)[0]
            print(inputs_ids)
            output_text = self.llama_tokenizer.decode(output_text[len(inputs_ids):])
            print(prompt, output_text, self.llama_tokenizer.eos_token)
            output_text = output_text.split(self.llama_tokenizer.eos_token)[0]
        else:
            style_kw_list = []
            for attr in attributes:
                prompt = STYLE_PROMPT_ATTRIBUTE[scene]
                prompt = prompt.replace('<Keyword_num/>', str(self.style_keyword_num))
                prompt = prompt.replace('<Items/>', items_text)
                prompt = prompt.replace('<Attribute/>', attr)
                inputs_ids = self.llama_tokenizer(prompt, max_length=self.llama_config.seq_length, padding="max_length")["input_ids"]
                output_text = self.llama_model.generate(inputs_ids,
                                                        max_new_tokens=self.llama_config.max_new_tokens,
                                                        do_sample=self.llama_config.do_sample,
                                                        top_k=self.llama_config.top_k,
                                                        top_p=self.llama_config.top_p)[0]
                output_text = self.llama_tokenizer.decode(output_text[len(inputs_ids):])
                output_text = output_text.split(self.llama_tokenizer.eos_token)[0]
                style_kw = re.findall(r'\d\. ([a-zA-Z\' ]*)', output_text)
                style_kw = [x.strip() for x in style_kw]
                style_kw_list += [x for x in style_kw if x]
                inputs_ids = self.llama_tokenizer(prompt, max_length=self.llama_config.seq_length, padding="max_length")["input_ids"]
                output_text = self.llama_model.generate(inputs_ids,
                                                        max_new_tokens=self.llama_config.max_new_tokens,
                                                        do_sample=self.llama_config.do_sample,
                                                        top_k=self.llama_config.top_k,
                                                        top_p=self.llama_config.top_p)[0]
                output_text = self.llama_tokenizer.decode(output_text[len(inputs_ids):])
                output_text = output_text.split(self.llama_tokenizer.eos_token)[0]
            
            prompt = STYLE_PROMPT_ATTRIBUTE_INTEGRATE[scene]
            prompt = prompt.replace('<Keyword_num/>', str(self.style_keyword_num))
            prompt = prompt.replace('<Items/>', items_text)
            prompt = prompt.replace('<Candidate_keywords/>', ','.join(style_kw_list))
                
        style_kw = re.findall(r'\d\. ([a-zA-Z\' ]*)', output_text)
        style_kw = [x.strip() for x in style_kw]
        style_kw = [x for x in style_kw if x]
        return style_kw
    
    def stylePromptSoft(self, scene, item_list):
        items_text = ' '.join(['%d. %s'%(i, self.itemProcess(scene, item)) for i, item in enumerate(item_list)])
        prompt = STYLE_PROMPT_SOFT[scene]
        prompt = prompt.replace('<Keyword_num/>', str(self.style_keyword_num))
        prompt = prompt.replace('<Items/>', items_text)
        # print(prompt)
        tokens = self.llama_tokenizer(prompt, return_tensors="ms",).input_ids[0].tolist()
        token_len = ms.Tensor([len(tokens)])
        tokens += [self.llama_tokenizer.pad_token_id] * self.num_image_prompt
        tokens = ms.Tensor([tokens]).cuda(self.cuda_id)
        image_emb = self.model[scene].forward(self.llama_tokenizer, self.llama_model, tokens, token_len)
        return image_emb