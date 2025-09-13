# DMFNER: A New Descriptor for a Diffusion-based Multi-feature Extraction and Fusion Method for Named Entity Recognition

Code for the paper *“DMFNER: A New Descriptor for a Diffusion-based Multi-feature Extraction and Fusion Method for Named Entity Recognition”

---

##  Setup

To run the code, you need:

```bash
conda create -n DMFNER python=3.8
pip install -r requirements.txt
```

##  Datasets

Nested NER:
- ACE04: https://catalog.ldc.upenn.edu/LDC2005T09
- GENIA: http://www.geniaproject.org/genia-corpuss

Flat NER:
- CoNLL03: https://data.deepai.org/conll2003.zip
- MSRA: https://github.com/ShannonAI/mrc-for-flat-nested-ner/blob/master/ner2mrc/download.md
- WNUT17: https://github.com/leondz/emerging_entities_17

We provide the preprocessed datasets in these links:
[ACE2004]([https://link-to-ace-checkpoint](https://drive.google.com/drive/folders/19mi-R8FbMwSRa0QqQ2sraTBA3vzZ5Shb?usp=drive_link)), 
[GENIA]([https://link-to-genia-checkpoint](https://drive.google.com/drive/folders/1qhMyrWzSw5yCtiQN3LA6hKdAP7jLnH2L?usp=drive_link)), 
[CoNLL03]([https://link-to-conll03-checkpoint](https://drive.google.com/drive/folders/1QnOdSs7l_gs5CBue9OXPF1OVX9zmjd7U?usp=drive_link)), 
[MSRA]([https://link-to-msra-checkpoint](https://drive.google.com/drive/folders/1gCEyRy4zPEgVPcqH3c2Ug7Wp7RfLa6E6?usp=drive_link)). . Please download them and put them into the data/datasets folder.

##  Training
```
python DMFNER.py train --config configs/wnut17.conf
```
