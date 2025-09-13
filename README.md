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

##  Training
```
python DMFNER.py train --config configs/wnut17.conf
```
