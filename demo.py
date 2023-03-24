from __future__ import print_function


from src.misc.config import cfg, cfg_from_file
from src.dataset import TextDataset
from src.trainer import condGANTrainer as trainer

import time
import random
import pprint
import numpy as np

import torch
import torchvision.transforms as transforms

from pathlib import Path
import streamlit as st


def gen_example(wordtoix, algo, text):
    """generate images from example sentences"""
    from nltk.tokenize import RegexpTokenizer

    data_dic = {}

    captions = []
    cap_lens = []

    sent = text.replace("\ufffd\ufffd", " ")
    tokenizer = RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(sent.lower())

    rev = []
    for t in tokens:
        t = t.encode("ascii", "ignore").decode("ascii")
        if len(t) > 0 and t in wordtoix:
            rev.append(wordtoix[t])
    captions.append(rev)
    cap_lens.append(len(rev))
    max_len = np.max(cap_lens)

    sorted_indices = np.argsort(cap_lens)[::-1]
    cap_lens = np.asarray(cap_lens)
    cap_lens = cap_lens[sorted_indices]
    cap_array = np.zeros((len(captions), max_len), dtype="int64")
    for i in range(len(captions)):
        idx = sorted_indices[i]
        cap = captions[idx]
        c_len = len(cap)
        cap_array[i, :c_len] = cap
    name = "output"
    key = name[(name.rfind("/") + 1) :]
    data_dic[key] = [cap_array, cap_lens, sorted_indices]
    algo.gen_example(data_dic)


# streamlit function


def center_element(type, text=None, img_path=None):
    """
    Function to center a streamlit element (text, image, etc)
    """
    if type == "image":
        col1, col2, col3 = st.columns([1, 2, 1])

    elif type == "text" or type == "heading":
        col1, col2, col3 = st.columns([1, 6, 1])

    elif type == "subheading":
        col1, col2, col3 = st.columns([1, 2, 1])

    elif type == "title":
        col1, col2, col3 = st.columns([1, 8, 1])

    with col1:
        st.write("")

    with col2:
        if type == "heading":
            st.header(text)

        elif type == "title":
            st.title(text)

        elif type == "image":
            st.image(img_path)

        elif type == "text":
            st.write(text)

        elif type == "subheading":
            st.subheader(text)

        # else:
        #     raise Exception("Unsupported input type")

    with col3:
        st.write("")


def demo_gan():

    cfg_from_file("eval_bird.yml")
    # print("Using config:")
    # pprint.pprint(cfg)
    cfg.CUDA = False
    manualSeed = 100
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    output_dir = "output/"
    split_dir = "test"
    bshuffle = True
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose(
        [
            transforms.Resize(int(imsize * 76 / 64)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip(),
        ]
    )
    st.cache(func=TextDataset, persist=True,ttl=10000)
    dataset = TextDataset(
        cfg.DATA_DIR, split_dir, base_size=cfg.TREE.BASE_SIZE, transform=image_transform
    )
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=True,
        shuffle=bshuffle,
        num_workers=int(cfg.WORKERS),
    )

    # Define models and go to train/evaluate
    st.cache(
        func=trainer, persist=True, suppress_st_warning=True,ttl=10000
    )

    algo = trainer(output_dir, dataloader, dataset.n_words, dataset.ixtoword)

    st.title("Text-To-Image Generation using AttnGAN")

    # st.markdown("---")
    # # st.markdown("Creator: [Atharva Ingle](https://github.com/Gladiator07)")
    # # st.markdown(
    # #     "Code: [GitHub Repository](https://github.com/Gladiator07/Text-to-image-synthesis-with-AttnGAN)"
    # # )
    # st.markdown("---")

    st.subheader("Enter the description of the bird in the text box !")
    # st.write(
    #     "**Example**: A yellow bird with red crown, black short beak and long tail."
    # )
    # st.markdown("**PS**: The synthesized birds might not even exist on earth 😬 😮 😱")
    st.markdown("#")
    selected_caption = st.selectbox( 'Select The Caption', 
    ('a large gray bird with a long wingspan and a long black beak.',
     'this large bird is solid dark gray color with a bit of white mixed in on its belly.',
      'a large bird with a long grey tail and black wings with bird head',
      'this bird has a tan underbelly with blue wings and a blue black.',
      'a small sized bird that has a white belly and black head.',
      'a bird with a small poitned bill, white eyebrows, and fluffy white and brown breast.'))

    user_input = st.text_input("Write the bird description below:", selected_caption)
    st.markdown("---")

    if user_input:

        start_t = time.time()

        # generate images for customized captions
        gen_example(dataset.wordtoix, algo, text=user_input)
        end_t = time.time()
        print("Total time for training:", end_t - start_t)
        st.write(f"**Your input**: {user_input}")
        center_element(type="subheading", text="AttnGAN synthesized bird")
        st.text("")
        center_element(
            type="image", img_path="models/bird_AttnGAN2/output/0_s_0_g2.png"
        )

        center_element(type="subheading", text="The attention given for each word in Image Description")
        st.image("models/bird_AttnGAN2/output/0_s_0_a1.png")

        st.markdown("---")
        with st.expander("Click to see the first stage images"):
            st.write("First stage image")
            st.image("models/bird_AttnGAN2/output/0_s_0_g1.png")
            st.write("First stage attention")
            st.image("models/bird_AttnGAN2/output/0_s_0_a0.png")




