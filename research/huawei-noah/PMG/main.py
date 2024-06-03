import numpy as np
import gradio as gr
import os
from functools import partial
from PIL import Image
from mindspore import Tensor, set_seed
import json
from inference import PMGInference
from PMG_utils import loadTestData, loadFullData, loadLlama, loadSDPipeline



# Scenes = ['emoticon', 'movie', 'costume']
Scenes = ['movie']
Attributes = {'costume': ['color', 'material', 'season', 'style', 'elements'],
              'movie': ['country', 'style', 'genre', 'age', 'actor', 'diretor']}
def loadDemoData():
    target_item_list = {}
    user_list = {}
    item_data = {}
    image_data = {}
    target_item_data = {}
    
    for scene in Scenes:
        path = os.path.join('data', scene)
        with open(os.path.join(path, 'items.txt'), 'r') as f:
            target_item_list[scene] = [x.strip() for x in f.readlines()]
        with open(os.path.join(path, 'users.txt'), 'r') as f:
            user_list[scene] = []
            for line in f.readlines():
                name, items = line.split(':')
                user_list[scene].append({'name':name.strip(), 'items': [x.strip() for x in items.split(';')]})
    
        _, item_data[scene], image_data[scene] = loadFullData(dataset=scene)
        
        if scene == 'emoticon':
            with open(os.path.join(path, 'conversations.json'), 'r') as f:
                target_item_data[scene] = json.load(f)
        else:
            target_item_data[scene] = item_data[scene]
    
    return target_item_list, user_list, item_data, image_data, target_item_data

target_item_list, user_list, item_data, image_data, target_item_data = loadDemoData()

item_name_list = {}
user_name_list = {}
item_name_dict = {}
user_name_dict = {}
for scene in Scenes:
    item_name_list[scene] = [ target_item_data[scene][x]['name'] for x in target_item_list[scene] ]
    user_name_list[scene] = [ x['name'] for x in user_list[scene] ]
    item_name_dict[scene] = { target_item_data[scene][x]['name']:x for x in target_item_list[scene] }
    user_name_dict[scene] = { x:i for i, x in enumerate(user_name_list[scene]) }
inference = PMGInference()

def userSelectionListener(scene, user_selection):
    user_id = user_name_dict[scene][user_selection]
    return [image_data[scene][user_list[scene][user_id]['items'][i]] for i in range(4)]

def targetItemSelectionListener(scene, target_item_selection):
    item_id = item_name_dict[scene][target_item_selection]
    if scene == 'emoticon':
        return preprocessChat(target_item_data[scene][item_id]['conversation'])
    else:
        return image_data[scene][item_id]

def generateKeywords(scene, user_selection, target_item_selection, seed):
    print('generateKeywords start', seed)
    global g_style_kw, g_item_kw
    set_seed(int(seed))
    user_id = user_name_dict[scene][user_selection]
    item_id = item_name_dict[scene][target_item_selection]
    item_list = [item_data[scene][item_id] for item_id in user_list[scene][user_id]['items']]
    g_item_kw = []
    if scene == 'movie':
        style_kw = inference.stylePrompt(scene, item_list)
        target_item = item_data[scene][item_id]
        item_kw = ['named \'{}\''.format(target_item['name'])] + target_item['keyword'][:5]
        g_item_kw += ['movie poster', target_item['name']]
    elif scene == 'costume':
        style_kw = inference.stylePrompt(scene, item_list)
        target_item = item_data[scene][item_id]
        item_kw = target_item['keywords'][:5]
        g_item_kw += ['costume']
    elif scene == 'emoticon':
        style_kw = inference.generateCharacter(item_list)
        item_kw = inference.generateEmoKeywords(target_item_data[scene][item_id]['conversation'])
        g_item_kw += ['A cute emoji, stickers, biaoqing, White background']
        
    style_kw = [x.strip() for x in style_kw]
    item_kw = [x.strip() for x in item_kw]
    g_style_kw = style_kw
    g_item_kw += item_kw
    return ', '.join(style_kw), ', '.join(item_kw)

def generateImage(scene, user_selection, target_item_selection, seed, inference_steps, w_soft_emb, style_w, num):
    global g_style_kw, g_item_kw
    user_id = user_name_dict[scene][user_selection]
    item_list = [item_data[scene][item_id] for item_id in user_list[scene][user_id]['items']]
    soft_emb = None
    gen_images = []
    style_weight_range = range(1, 3)
    if scene == 'costume':
        style_weight_range = range(2, 4)
    elif scene == 'emoticon':
        style_weight_range = range(1, 2)
    
    # print(item_list, g_style_kw, g_item_kw)
    for i in style_weight_range:
        gen_images += inference.generateImage(scene,
                                              item_kw=g_item_kw,
                                              style_kw=g_style_kw,
                                              soft_prompt=soft_emb,
                                              num=int(num),
                                              style_weight=i,
                                              seed=int(seed),
                                              inference_steps=int(inference_steps))
    best_img_id = inference.calcBestImage(scene, ', '.join(g_item_kw), ', '.join(g_style_kw), gen_images)
    return gen_images[best_img_id]

def preprocessChat(conv):
    if len(conv)%2 == 1:
        conv = [None] + conv
    return [ [conv[i], conv[i+1]] for i in range(0, len(conv), 2) ]

if __name__ == "__main__":
    with open('./main.css', 'r') as f:
        css = f.read()
    
    with gr.Blocks(css=css, title='PMG') as demo:
        for scene in Scenes:
            with gr.Tab(scene.capitalize()):
                with gr.Row():
                    with gr.Column(elem_id="input"): #Inputs
                        gr.Markdown("# Inputs")
                        with gr.Column(scale=0, elem_id="input_user"):
                            gr.Markdown("## User")
                            user_selection = gr.Dropdown(user_name_list[scene], value=0, label="User preference")
                                
                            if scene == 'movie':
                                history_item_label = "historical watched movies"
                            elif scene == 'emoticon':
                                history_item_label = "historical used emoticons"
                            gr.Markdown( 'User behavior ({}) :'.format(history_item_label))
                            with gr.Row(elem_id='history_item'):
                                # user_item_shortcut = gr.Gallery(type="pil",
                                #                                 value=[image_data[scene][user_list[scene][0]['items'][i]] for i in range(4)],
                                #                                 label=history_item_label,
                                #                                 columns=[2],
                                #                                 show_download_button=False)
                                user_item_shortcut = []
                                user_item_name = []
                                for i in range(4):
                                    item_id = user_list[scene][0]['items'][i]
                                    with gr.Column(min_width=100):
                                        user_item_shortcut.append(gr.Image(type="pil",
                                                                           value=image_data[scene][item_id],
                                                                           min_width=100,
                                                                           show_label=False,
                                                                           show_download_button=False))
                        
                        with gr.Column(scale=0, elem_id="input_scenario"):
                            gr.Markdown("## Scenario")
                            with gr.Row(elem_id='target_item_selection'):
                                target_item_label = "Target item"
                                if scene == "emoticon":
                                    target_item_label = "Target conversation"
                                elif scene == "movie":
                                    target_item_label = "Target movie"
                                target_item_selection = gr.Dropdown(item_name_list[scene], value=0, label=target_item_label)
                                default = target_item_list[scene][0]
                                if scene == 'emoticon':
                                    target_item_shortcut = gr.Chatbot(type="pil", value=preprocessChat(target_item_data[scene][default]['conversation']))
                                else:
                                    target_item_shortcut = gr.Image(type="pil", value=image_data[scene][default], show_download_button=False, show_label=False)
                        
                        with gr.Accordion("Advanced Options", scale=0, open=False):
                            seed = gr.Textbox(label='Random seed', value=1526, show_label=True)
                            inference_steps = gr.Slider(minimum=10, maximum=100, value=30, step=1, label='Diffusion inference step number', show_label=True)
                            soft_checkbox = gr.Checkbox(label='with soft embedding', value=(scene != 'emoticon'), show_label=True, visible=False)
                            num = gr.Textbox(label='image number', value=1, show_label=True, visible=False)
                            style_w = gr.Textbox(label='preference keywords weight', value=0.2, show_label=True, visible=False)
                            
                        generate_button = gr.Button("Run")
                    
                    with gr.Column(elem_id="output"): #Outputs
                        gr.Markdown("# Outputs")
                        with gr.Column(scale=0, elem_id="output_keyword"):
                            gr.Markdown("## Step 1: Keywords Generation")
                            gr.Markdown("Preference keywords are generated according to the user's beheavior. Target item keywords are generated according to the typical scenario.")
                            style_keywords_text = 'Preference keywords'
                            target_item_keywords_text = 'Target item keywords'
                            
                            if scene == 'emoticon':
                                style_keywords_text = 'Preference keywords (In emoticon generation, we design the character in emoticon that match user perference)'
                                # target_item_keywords_text = 'Emotion'
                                
                            style_keywords = gr.Textbox(label=style_keywords_text, interactive=False, show_label=True)
                            target_item_keywords = gr.Textbox(label=target_item_keywords_text, interactive=False, show_label=True)
                        
                        with gr.Column(scale=0, elem_id="output_image"):
                            gr.Markdown("## Step 2: Image Generation")
                            gr.Markdown("The response image is generated by weighted combining perference keywords, soft embeddings and target item keywords as prompts in stable diffusion.")
                            generation = gr.Image(type="pil", elem_classes='image_class', show_label=False)

                user_selection.change(partial(userSelectionListener, scene), inputs=user_selection, outputs=user_item_shortcut)
                target_item_selection.change(partial(targetItemSelectionListener, scene), inputs=target_item_selection, outputs=target_item_shortcut)
                
                generate_button.click(
                    partial(generateKeywords, scene),
                    inputs=[user_selection, target_item_selection, seed],
                    outputs=[style_keywords, target_item_keywords]
                ).then(
                    partial(generateImage, scene),
                    inputs=[user_selection, target_item_selection, seed, inference_steps, soft_checkbox, style_w, num],
                    outputs=generation
                )

    demo.launch(debug=True, ssl_verify=False)