# Awesome Scientific Language Models
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**NOTE 1**: To avoid ambiguity, when we talk about the number of parameters in a model, "Base" refers to 110M (i.e., BERT-Base) and "Large" refers to 340M (i.e., BERT-Large). Other numbers will be written explicitly.

**NOTE 2**: In each subsection, papers are sorted chronologically. If a paper has a preprint (e.g., arXiv or bioRxiv) version, its publication date is according to the preprint service. Otherwise, its publication date is according to the conference proceeding or journal.


## General
### Language
**(SciBERT)** _SciBERT: A Pretrained Language Model for Scientific Text_ ```EMNLP 2019```     
[[Paper](https://arxiv.org/abs/1903.10676)] [[GitHub](https://github.com/allenai/scibert)] [[Model (Base)](https://huggingface.co/allenai/scibert_scivocab_uncased)]

**(SciGPT2)** _Explaining Relationships Between Scientific Documents_ ```ACL 2021```     
[[Paper](https://arxiv.org/abs/2002.00317)] [[GitHub](https://github.com/Kel-Lu/SciGen)] [[Model (117M)](https://drive.google.com/file/d/1AoNYnhvI6tensnrpQVc09KL1NWJ5MvFU/view)]

**(CATTS)** _TLDR: Extreme Summarization of Scientific Documents_ ```EMNLP 2020 Findings```     
[[Paper](https://arxiv.org/abs/2004.15011)] [[GitHub](https://github.com/allenai/scitldr)] [[Model (406M)](https://storage.cloud.google.com/skiff-models/scitldr/catts-xsum.tldr-aic.pt)]

**(SciNewsBERT)** _SciClops: Detecting and Contextualizing Scientific Claims for Assisting Manual Fact-Checking_ ```CIKM 2021```     
[[Paper](https://arxiv.org/abs/2110.13090)] [[Model (Base)](https://huggingface.co/psmeros/SciNewsBERT)]

**(ScholarBERT)** _The Diminishing Returns of Masked Language Models to Science_ ```ACL 2023 Findings```     
[[Paper](https://arxiv.org/abs/2205.11342)] [[Model (Large)](https://huggingface.co/globuslabs/ScholarBERT)] [[Model (770M)](https://huggingface.co/globuslabs/ScholarBERT-XL)]

**(Galactica)** _Galactica: A Large Language Model for Science_ ```arXiv 2022```     
[[Paper](https://arxiv.org/abs/2211.09085)]


### Graph-Enhanced
**(SPECTER)** _SPECTER: Document-level Representation Learning using Citation-informed Transformers_ ```ACL 2020```     
[[Paper](https://arxiv.org/abs/2004.07180)] [[GitHub](https://github.com/allenai/specter)] [[Model (Base)](https://huggingface.co/allenai/specter)]

**(OAG-BERT)** _OAG-BERT: Towards a Unified Backbone Language Model for Academic Knowledge Services_ ```KDD 2022```     
[[Paper](https://arxiv.org/abs/2103.02410)] [[GitHub](https://github.com/THUDM/OAG-BERT)]

**(ASPIRE)** _Multi-Vector Models with Textual Guidance for Fine-Grained Scientific Document Similarity_ ```NAACL 2022```     
[[Paper](https://arxiv.org/abs/2111.08366)] [[GitHub](https://github.com/allenai/aspire)] [[Model (Base)](https://huggingface.co/allenai/aspire-contextualsentence-multim-compsci)]

**(SciNCL)** _Neighborhood Contrastive Learning for Scientific Document Representations with Citation Embeddings_ ```EMNLP 2022```     
[[Paper](https://arxiv.org/abs/2202.06671)] [[GitHub](https://github.com/malteos/scincl)] [[Model (Base)](https://huggingface.co/malteos/scincl)]

**(SPECTER 2.0)** _SciRepEval: A Multi-Format Benchmark for Scientific Document Representations_ ```EMNLP 2023```     
[[Paper](https://arxiv.org/abs/2211.13308)] [[GitHub](https://github.com/allenai/SPECTER2)] [[Model (113M)](https://huggingface.co/allenai/specter2)]

**(SciMult)** _Pre-training Multi-task Contrastive Learning Models for Scientific Literature Understanding_ ```EMNLP 2023 Findings```     
[[Paper](https://arxiv.org/abs/2305.14232)] [[GitHub](https://github.com/yuzhimanhua/SciMult)] [[Model (138M)](https://huggingface.co/yuz9yuz/SciMult)]



## Biology and Medicine
**Acknowledgment: We referred to Wang et al.'s survey paper [_Pre-trained Language Models in Biomedical Domain: A
Systematic Survey_](https://arxiv.org/abs/2110.05006) when writing some parts of this section.**

### Language
**(BioBERT)** _BioBERT: A Pre-trained Biomedical Language Representation Model for Biomedical Text Mining_ ```Bioinformatics 2020```     
[[Paper](https://arxiv.org/abs/1901.08746)] [[GitHub](https://github.com/dmis-lab/biobert)] [[Model (Base)](https://huggingface.co/dmis-lab/biobert-base-cased-v1.2)] [[Model (Large)](https://huggingface.co/dmis-lab/biobert-large-cased-v1.1)]

**(BioELMo)** _Probing Biomedical Embeddings from Language Models_ ```NAACL 2019 Workshop```     
[[Paper](https://arxiv.org/abs/1904.02181)] [[GitHub](https://github.com/Andy-jqa/bioelmo)] [[Model (93M)](https://drive.google.com/file/d/1BQIuWGoZDVWppiz9Cst-ZqWd2mLiY2nc/view)]

**(Bio+Clinical BERT)** _Publicly Available Clinical BERT Embeddings_ ```NAACL 2019 Workshop```     
[[Paper](https://arxiv.org/abs/1904.03323)] [[GitHub](https://github.com/EmilyAlsentzer/clinicalBERT)] [[Model (Base)](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)]

**(ClinicalBERT)** _ClinicalBERT: Modeling Clinical Notes and Predicting Hospital Readmission_ ```CHIL 2020 Workshop```     
[[Paper](https://arxiv.org/abs/1904.05342)] [[GitHub](https://github.com/kexinhuang12345/clinicalBERT)] [[Model (Base)](https://drive.google.com/file/d/1X3WrKLwwRAVOaAfKQ_tkTi46gsPfY5EB/edit)]

**(BlueBERT, f.k.a. NCBI-BERT)** _Transfer Learning in Biomedical Natural Language Processing: An Evaluation of BERT and ELMo on Ten Benchmarking Datasets_ ```ACL 2019 Workshop```     
[[Paper](https://arxiv.org/abs/1906.05474)] [[GitHub](https://github.com/ncbi-nlp/bluebert)] [[Model (Base)](https://huggingface.co/bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12)] [[Model (Large)](https://huggingface.co/bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16)]

**(BEHRT)** _BEHRT: Transformer for Electronic Health Records_ ```Scientific Reports 2020```     
[[Paper](https://arxiv.org/abs/1907.09538)] [[GitHub](https://github.com/deepmedicine/BEHRT)]

**(EhrBERT)** _Fine-Tuning Bidirectional Encoder Representations From Transformers (BERT)–Based Models on Large-Scale Electronic Health Record Notes: An Empirical Study_ ```JMIR Medical Informatics 2019```     
[[Paper](https://medinform.jmir.org/2019/3/e14830)] [[GitHub](https://github.com/umassbento/ehrbert)]

**(Clinical XLNet)** _Clinical XLNet: Modeling Sequential Clinical Notes and Predicting Prolonged Mechanical Ventilation_ ```EMNLP 2020 Workshop```     
[[Paper](https://arxiv.org/abs/1912.11975)] [[GitHub](https://github.com/lindvalllab/clinicalXLNet)]

**(ouBioBERT)** _Pre-training Technique to Localize Medical BERT and Enhance Biomedical BERT_ ```arXiv 2020```     
[[Paper](https://arxiv.org/abs/2005.07202)] [[GitHub](https://github.com/sy-wada/blue_benchmark_with_transformers)] [[Model (Base)](https://huggingface.co/seiya/oubiobert-base-uncased)]

**(COVID-Twitter-BERT)** _COVID-Twitter-BERT: A Natural Language Processing Model to Analyse COVID-19 Content on Twitter_ ```Frontiers in Artificial Intelligence 2023```     
[[Paper](https://arxiv.org/abs/2005.07503)] [[GitHub](https://github.com/digitalepidemiologylab/covid-twitter-bert)] [[Model (Large)](https://huggingface.co/digitalepidemiologylab/covid-twitter-bert-v2)]

**(Med-BERT)** _Med-BERT: Pretrained Contextualized Embeddings on Large-Scale Structured Electronic Health Records for Disease Prediction_ ```npj Digital Medicine 2021```     
[[Paper](https://arxiv.org/abs/2005.12833)] [[GitHub](https://github.com/ZhiGroup/Med-BERT)]

**(Bio-ELECTRA)** _On the Effectiveness of Small, Discriminatively Pre-trained Language Representation Models for Biomedical Text Mining_ ```EMNLP 2020 Workshop```     
[[Paper](https://www.biorxiv.org/content/10.1101/2020.05.20.107003)] [[GitHub](https://github.com/SciCrunch/bio_electra)] [[Model (Base)](https://zenodo.org/records/3971235)]

**(BiomedBERT, f.k.a. PubMedBERT)** _Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing_ ```ACM Transactions on Computing for Healthcare 2021```     
[[Paper](https://arxiv.org/abs/2007.15779)] [[Model (Base)](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract)] [[Model (Large)](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract)]

**(MCBERT)** _Conceptualized Representation Learning for Chinese Biomedical Text Mining_ ```arXiv 2020```     
[[Paper](https://arxiv.org/abs/2008.10813)] [[GitHub](https://github.com/alibaba-research/ChineseBLUE)] [[Model (Base)](https://drive.google.com/file/d/1ccXRvaeox5XCNP_aSk_ttLBY695Erlok/view)]

**(BRLTM)** _Bidirectional Representation Learning From Transformers Using Multimodal Electronic Health Record Data to Predict Depression_ ```JBHI 2021```     
[[Paper](https://arxiv.org/abs/2009.12656)] [[GitHub](https://github.com/lanyexiaosa/brltm)]

**(BioRedditBERT)** _COMETA: A Corpus for Medical Entity Linking in the Social Media_ ```EMNLP 2020```     
[[Paper](https://arxiv.org/abs/2010.03295)] [[GitHub](https://github.com/cambridgeltl/cometa)] [[Model (Base)](https://huggingface.co/cambridgeltl/BioRedditBERT-uncased)]

**(BioMegatron)** _BioMegatron: Larger Biomedical Domain Language Model_ ```EMNLP 2020```     
[[Paper](https://arxiv.org/abs/2010.06060)] [[GitHub](https://github.com/NVIDIA/NeMo)] [[Model (Large)](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/biomegatron345m_biovocab_50k_uncased)]

**(SapBERT)** _Self-Alignment Pretraining for Biomedical Entity Representations_ ```NAACL 2021```     
[[Paper](https://arxiv.org/abs/2010.11784)] [[GitHub](https://github.com/cambridgeltl/sapbert)] [[Model (Base)](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext)]

**(ClinicalTransformer)** _Clinical Concept Extraction Using Transformers_ ```JAMIA 2020```     
[[Paper](https://academic.oup.com/jamia/article-abstract/27/12/1935/5943218)] [[GitHub](https://github.com/uf-hobi-informatics-lab/ClinicalTransformerNER)] [[Model (BERT)](https://transformer-models.s3.amazonaws.com/mimiciii_bert_10e_128b.zip)] [[Model (RoBERTa)](https://transformer-models.s3.amazonaws.com/mimiciii_roberta_10e_128b.zip)] [[Model (ALBERT)](https://transformer-models.s3.amazonaws.com/mimiciii_albert_10e_128b.zip)] [[Model (ELECTRA)](https://transformer-models.s3.amazonaws.com/mimiciii_electra_5e_128b.zip)] [[Model (XLNet)](https://transformer-models.s3.amazonaws.com/mimiciii_xlnet_5e_128b.zip)] [[Model (Longformer)](https://transformer-models.s3.amazonaws.com/mimiciii_longformer_5e_128b.zip)] [[Model (DeBERTa)](https://transformer-models.s3.amazonaws.com/mimiciii_deberta_10e_128b.tar.gz)]

**(BioRoBERTa)** _Pretrained Language Models for Biomedical and Clinical Tasks: Understanding and Extending the State-of-the-Art_ ```EMNLP 2020 Workshop```     
[[Paper](https://aclanthology.org/2020.clinicalnlp-1.17)] [[GitHub](https://github.com/facebookresearch/bio-lm)] [[Model (125M)](https://dl.fbaipublicfiles.com/biolm/RoBERTa-base-PM-M3-Voc-train-longer-hf.tar.gz)] [[Model (355M)](https://dl.fbaipublicfiles.com/biolm/RoBERTa-large-PM-M3-Voc-hf.tar.gz)]

**(RAD-BERT)** _Highly Accurate Classification of Chest Radiographic Reports Using a Deep Learning Natural Language Model Pre-trained on 3.8 Million Text Reports_ ```Bioinformatics 2020```     
[[Paper](https://academic.oup.com/bioinformatics/article/36/21/5255/5875602)] [[GitHub](https://github.com/rAIdiance/bert-for-radiology)]

**(BioMedBERT)** _BioMedBERT: A Pre-trained Biomedical Language Model for QA and IR_ ```COLING 2020```     
[[Paper](https://aclanthology.org/2020.coling-main.59)] [[GitHub](https://github.com/BioMedBERT/biomedbert)]

**(LBERT)** _LBERT: Lexically Aware Transformer-Based Bidirectional Encoder Representation Model for Learning Universal Bio-Entity Relations_ ```Bioinformatics 2021```     
[[Paper](https://academic.oup.com/bioinformatics/article/37/3/404/5893949)] [[GitHub](https://github.com/warikoone/LBERT)]

**(ELECTRAMed)** _ELECTRAMed: A New Pre-trained Language Representation Model for Biomedical NLP_ ```arXiv 2021```     
[[Paper](https://arxiv.org/abs/2104.09585)] [[GitHub](https://github.com/gmpoli/electramed)] [[Model (Base)](https://huggingface.co/giacomomiolo/electramed_base_scivocab_1M)]

**(SciFive)** _SciFive: A Text-to-Text Transformer Model for Biomedical Literature_ ```arXiv 2021```     
[[Paper](https://arxiv.org/abs/2106.03598)] [[GitHub](https://github.com/justinphan3110/SciFive)] [[Model (220M)](https://huggingface.co/razent/SciFive-base-Pubmed_PMC)] [[Model (770M)](https://huggingface.co/razent/SciFive-large-Pubmed_PMC)]

**(BioALBERT)** _Benchmarking for Biomedical Natural Language Processing Tasks with a Domain Specific ALBERT_ ```BMC Bioinformatics 2022```     
[[Paper](https://arxiv.org/abs/2107.04374)] [[GitHub](https://github.com/usmaann/BioALBERT)] [[Model (12M)](https://drive.google.com/file/d/1SIBd_-GETHhMiZ7BgMdDPEUDjOjtN_bH/view)] [[Model (18M)](https://drive.google.com/file/d/16KRtHf8Meze2Hcc4vK_GUNhG-9LY6_6P/view)]

**(Clinical-Longformer)** _Clinical-Longformer and Clinical-BigBird: Transformers for Long Clinical Sequences_ ```arXiv 2021```     
[[Paper](https://arxiv.org/abs/2201.11838)] [[GitHub](https://github.com/luoyuanlab/Clinical-Longformer)] [[Model (Longformer)](https://huggingface.co/yikuan8/Clinical-Longformer)] [[Model (BigBird)](https://huggingface.co/yikuan8/Clinical-BigBird)]

**(BioBART)** _BioBART: Pretraining and Evaluation of A Biomedical Generative Language Model_ ```ACL 2022 Workshop```     
[[Paper](https://arxiv.org/abs/2204.03905)] [[GitHub](https://github.com/GanjinZero/BioBART)] [[Model (140M)](https://huggingface.co/GanjinZero/biobart-base)] [[Model (406M)](https://huggingface.co/GanjinZero/biobart-large)]

**(BioGPT)** _BioGPT: Generative Pre-trained Transformer for Biomedical Text Generation and Mining_ ```Briefings in Bioinformatics 2022```     
[[Paper](https://arxiv.org/abs/2210.10341)] [[GitHub](https://github.com/microsoft/BioGPT)] [[Model (355M)](https://huggingface.co/microsoft/biogpt)] [[Model (1.5B)](https://huggingface.co/microsoft/BioGPT-Large)]

**(BioMedLM, f.k.a. PubMedGPT)** _BioMedLM: a Domain-Specific Large Language Model for Biomedical Text_     
[[Blog](https://www.mosaicml.com/blog/introducing-pubmed-gpt)] [[GitHub](https://github.com/stanford-crfm/BioMedLM)] [[Model (2.7B)](https://huggingface.co/stanford-crfm/BioMedLM)]

**(Med-PaLM)** _Large Language Models Encode Clinical Knowledge_ ```Nature 2023```     
[[Paper](https://arxiv.org/abs/2212.13138)]

**(ChatDoctor)** _ChatDoctor: A Medical Chat Model Fine-Tuned on a Large Language Model Meta-AI (LLaMA) Using Medical Domain Knowledge_ ```Cureus 2023```     
[[Paper](https://arxiv.org/abs/2303.14070)] [[GitHub](https://github.com/Kent0n-Li/ChatDoctor)]

**(DoctorGLM)** _DoctorGLM: Fine-tuning your Chinese Doctor is not a Herculean Task_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2304.01097)] [[GitHub](https://github.com/xionghonglin/DoctorGLM)]

**(BenTsao, f.k.a. HuaTuo)** _HuaTuo: Tuning LLaMA Model with Chinese Medical Knowledge_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2304.06975)] [[GitHub](https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese)]

**(MedAlpaca)** _MedAlpaca - An Open-Source Collection of Medical Conversational AI Models and Training Data_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2304.08247)] [[GitHub](https://github.com/kbressem/medAlpaca)] [[Model (7B)](https://huggingface.co/medalpaca/medalpaca-7b)] [[Model (13B)](https://huggingface.co/medalpaca/medalpaca-13b)]

**(Med-PaLM 2)** _Towards Expert-Level Medical Question Answering with Large Language Models_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2305.09617)]

**(HuatuoGPT)** _HuatuoGPT, towards Taming Language Model to Be a Doctor_ ```EMNLP 2023 Findings```     
[[Paper](https://arxiv.org/abs/2305.15075)] [[GitHub](https://github.com/FreedomIntelligence/HuatuoGPT)] [[Model (7B)](https://huggingface.co/FreedomIntelligence/HuatuoGPT-7B)] [[Model (13B)](https://huggingface.co/FreedomIntelligence/HuatuoGPT-13b-delta)]

**(MedCPT)** _MedCPT: Contrastive Pre-trained Transformers with Large-scale PubMed Search Logs for Zero-shot Biomedical Information Retrieval_ ```Bioinformatics 2023```     
[[Paper](https://arxiv.org/abs/2307.00589)] [[GitHub](https://github.com/ncbi/MedCPT)] [[Model (Base)](https://huggingface.co/ncbi/MedCPT-Query-Encoder)]

**(DISC-MedLLM)** _DISC-MedLLM: Bridging General Large Language Models and Real-World Medical Consultation_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2308.14346)] [[GitHub](https://github.com/FudanDISC/DISC-MedLLM)] [[Model (13B)](https://huggingface.co/Flmc/DISC-MedLLM)]

**(HuatuoGPT-II)** _HuatuoGPT-II, One-stage Training for Medical Adaption of LLMs_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2311.09774)] [[GitHub](https://github.com/FreedomIntelligence/HuatuoGPT-II)] [[Model (7B)](https://huggingface.co/FreedomIntelligence/HuatuoGPT2-7B)] [[Model (13B)](https://huggingface.co/FreedomIntelligence/HuatuoGPT2-13B)] [[Model (34B)](https://huggingface.co/FreedomIntelligence/HuatuoGPT2-34B)]


### Graph-Enhanced
**(G-BERT)** _Pre-training of Graph Augmented Transformers for Medication Recommendation_ ```IJCAI 2019```     
[[Paper](https://arxiv.org/abs/1906.00346)] [[GitHub](https://github.com/jshang123/G-Bert)]

**(CODER)** _CODER: Knowledge Infused Cross-Lingual Medical Term Embedding for Term Normalization_ ```JBI 2022```     
[[Paper](https://arxiv.org/abs/2011.02947)] [[GitHub](https://github.com/GanjinZero/CODER)] [[Model (Base)](https://huggingface.co/GanjinZero/coder_eng)]

**(KeBioLM)** _Improving Biomedical Pretrained Language Models with Knowledge_ ```NAACL 2021 Workshop```     
[[Paper](https://arxiv.org/abs/2104.10344)] [[GitHub](https://github.com/GanjinZero/KeBioLM)] [[Model (155M)](https://drive.google.com/file/d/1kMbTsc9rPpBc-6ezEHjMbQLljW3SUWG9/edit)]

**(BioLinkBERT)** _LinkBERT: Pretraining Language Models with Document Links_ ```ACL 2022```     
[[Paper](https://arxiv.org/abs/2203.15827)] [[GitHub](https://github.com/michiyasunaga/LinkBERT)] [[Model (Base)](https://huggingface.co/michiyasunaga/BioLinkBERT-base)] [[Model (Large)](https://huggingface.co/michiyasunaga/BioLinkBERT-large)]

**(DRAGON)** _Deep Bidirectional Language-Knowledge Graph Pretraining_ ```NeurIPS 2022```     
[[Paper](https://arxiv.org/abs/2210.09338)] [[GitHub](https://github.com/michiyasunaga/dragon)] [[Model (360M)](https://nlp.stanford.edu/projects/myasu/DRAGON/models/biomed_model.pt)]


### Vision-Language
**(ConVIRT)** _Contrastive Learning of Medical Visual Representations from Paired Images and Text_ ```MLHC 2022```     
[[Paper](https://arxiv.org/abs/2010.00747)] [[GitHub](https://github.com/yuhaozhang/convirt)]

**(MedViLL)** _Multi-modal Understanding and Generation for Medical Images and Text via Vision-Language Pre-Training_ ```JBHI 2022```     
[[Paper](https://arxiv.org/abs/2105.11333)] [[GitHub](https://github.com/SuperSupermoon/MedViLL)] [[Model](https://drive.google.com/file/d/1shOQrOWbkIeUUsQN48fEP6wj0e266jOb/view)]

**(GLoRIA)** _GLoRIA: A Multimodal Global-Local Representation Learning Framework for Label-efficient Medical Image Recognition_ ```ICCV 2021```     
[[Paper](https://ieeexplore.ieee.org/document/9710099)] [[GitHub](https://github.com/marshuang80/gloria)]

**(LoVT)** _Joint Learning of Localized Representations from Medical Images and Reports_ ```ECCV 2022```     
[[Paper](https://arxiv.org/abs/2112.02889)] [[GitHub](https://github.com/philip-mueller/lovt)]

**(CvT2DistilGPT2)** _Improving Chest X-Ray Report Generation by Leveraging Warm Starting_ ```Artificial Intelligence in Medicine 2023```     
[[Paper](https://arxiv.org/abs/2201.09405)] [[GitHub](https://github.com/aehrc/cvt2distilgpt2)] [[Model](https://doi.org/10.25919/ng3g-aj81)]

**(BioViL)** _Making the Most of Text Semantics to Improve Biomedical Vision-Language Processing_ ```ECCV 2022```     
[[Paper](https://arxiv.org/abs/2204.09817)] [[GitHub](https://github.com/microsoft/hi-ml/tree/main/hi-ml-multimodal)]

**(LViT)** _LViT: Language meets Vision Transformer in Medical Image Segmentation_ ```TMI 2022```     
[[Paper](https://arxiv.org/abs/2206.14718)] [[GitHub](https://github.com/HUANGLIZI/LViT)]

**(M3AE)** _Multi-Modal Masked Autoencoders for Medical Vision-and-Language Pre-Training_ ```MICCAI 2022```     
[[Paper](https://arxiv.org/abs/2209.07098)] [[GitHub](https://github.com/zhjohnchan/M3AE)]

**(ARL)** _Align, Reason and Learn: Enhancing Medical Vision-and-Language Pre-training with Knowledge_ ```ACM MM 2022```     
[[Paper](https://arxiv.org/abs/2209.07118)] [[GitHub](https://github.com/zhjohnchan/ARL)]

**(CheXzero)** _Expert-Level Detection of Pathologies from Unannotated Chest X-ray Images via Self-Supervised Learning_ ```Nature Biomedical Engineering 2022```     
[[Paper](https://www.nature.com/articles/s41551-022-00936-9)] [[GitHub](https://github.com/rajpurkarlab/CheXzero)] [[Model](https://drive.google.com/drive/folders/1makFLiEMbSleYltaRxw81aBhEDMpVwno)]

**(MGCA)** _Multi-Granularity Cross-modal Alignment for Generalized Medical Visual Representation Learning_ ```NeurIPS 2022```     
[[Paper](https://arxiv.org/abs/2210.06044)] [[GitHub](https://github.com/HKU-MedAI/MGCA)]

**(MedCLIP)** _MedCLIP: Contrastive Learning from Unpaired Medical Images and Text_ ```EMNLP 2022```     
[[Paper](https://arxiv.org/abs/2210.10163)] [[GitHub](https://github.com/RyanWangZf/MedCLIP)]

**(BioViL-T)** _Learning to Exploit Temporal Structure for Biomedical Vision-Language Processing_ ```CVPR 2023```     
[[Paper](https://arxiv.org/abs/2301.04558)] [[GitHub](https://github.com/microsoft/hi-ml/tree/main/hi-ml-multimodal)] [[Model](https://huggingface.co/microsoft/BiomedVLP-BioViL-T)]

**(RGRG)** _Interactive and Explainable Region-guided Radiology Report Generation_ ```CVPR 2023```     
[[Paper](https://arxiv.org/abs/2304.08295)] [[GitHub](https://github.com/ttanida/rgrg)] [[Model](https://drive.google.com/file/d/1rDxqzOhjqydsOrITJrX0Rj1PAdMeP7Wy/view)]

**(Med-PaLM M)** _Towards Generalist Biomedical AI_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2307.14334)] [[GitHub](https://github.com/kyegomez/Med-PaLM)]


### Other Modalities (Protein, DNA)
**(ProGen)** _Large Language Models Generate Functional Protein Sequences across Diverse Families_ ```Nature Biotechnology 2023```     
[[Paper](https://www.nature.com/articles/s41587-022-01618-2)]

**(ProtTrans)** _ProtTrans: Towards Cracking the Language of Life's Code Through Self-Supervised Deep Learning and High Performance Computing_ ```TPAMI 2021```     
[[Paper](https://arxiv.org/abs/2007.06225)] [[GitHub](https://github.com/agemagician/ProtTrans)] [[Model (Base, BERT)](https://huggingface.co/Rostlab/prot_bert)] [[Model (12M, ALBERT)](https://huggingface.co/Rostlab/prot_albert)] [[Model (Base, XLNet)](https://huggingface.co/Rostlab/prot_xlnet)] [[Model (3B, T5)](https://huggingface.co/Rostlab/prot_t5_xl_uniref50)] [[Model (11B, T5)](https://huggingface.co/Rostlab/prot_t5_xxl_uniref50)]

**(DNABERT)** _DNABERT: Pre-trained Bidirectional Encoder Representations from Transformers Model for DNA-Language in Genome_ ```Bioinformatics 2021```     
[[Paper](https://www.biorxiv.org/content/10.1101/2020.09.17.301879)] [[GitHub](https://github.com/jerryji1993/DNABERT)] [[Model (Base)](https://drive.google.com/file/d/1BJjqb5Dl2lNMg2warsFQ0-Xvn1xxfFXC/view)]

**(ESM-1b)** _Biological Structure and Function Emerge from Scaling Unsupervised Learning to 250 Million Protein Sequences_ ```PNAS 2021```     
[[Paper](https://www.pnas.org/doi/10.1073/pnas.2016239118)] [[GitHub](https://github.com/facebookresearch/esm)] [[Model (650M)](https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt)]

**(ESM-1v)** _Language Models Enable Zero-Shot Prediction of the Effects of Mutations on Protein Function_ ```NeurIPS 2021```     
[[Paper](https://www.biorxiv.org/content/10.1101/2021.07.09.450648)] [[GitHub](https://github.com/facebookresearch/esm)] [[Model (650M)](https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_1.pt)]

**(ProteinBERT)** _ProteinBERT: A Universal Deep-Learning Model of Protein Sequence and Function_ ```Bioinformatics 2022```     
[[Paper](https://academic.oup.com/bioinformatics/article/38/8/2102/6502274)] [[GitHub](https://github.com/nadavbra/protein_bert)] [[Model (16M)](https://huggingface.co/GrimSqueaker/proteinBERT)]

**(ESM-IF1)** _Learning Inverse Folding from Millions of Predicted Structures_ ```ICML 2022```     
[[Paper](https://www.biorxiv.org/content/10.1101/2022.04.10.487779)] [[GitHub](https://github.com/facebookresearch/esm)] [[Model (124M)](https://dl.fbaipublicfiles.com/fair-esm/models/esm_if1_gvp4_t16_142M_UR50.pt)]

**(ProtGPT2)** _ProtGPT2 is a Deep Unsupervised Language Model for Protein Design_ ```Nature Communications 2022```     
[[Paper](https://www.nature.com/articles/s41467-022-32007-7)] [[Model (738M)](https://huggingface.co/nferruz/ProtGPT2)]

**(ProGen2)** _ProGen2: Exploring the Boundaries of Protein Language Models_ ```Cell Systems 2023```     
[[Paper](https://arxiv.org/abs/2206.13517)] [[GitHub](https://github.com/salesforce/progen)] [[Model (151M)](https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-small.tar.gz)] [[Model (764M)](https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-base.tar.gz)] [[Model (2.7B)](https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-large.tar.gz)] [[Model (6.4B)](https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-xlarge.tar.gz)]

**(ESM-2)** _Evolutionary-Scale Prediction of Atomic-Level Protein Structure with a Language Model_ ```Science 2023```     
[[Paper](https://www.biorxiv.org/content/10.1101/2022.07.20.500902)] [[GitHub](https://github.com/facebookresearch/esm)] [[Model (8M)](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t6_8M_UR50D.pt)] [[Model (35M)](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t12_35M_UR50D.pt)] [[Model (150M)](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t30_150M_UR50D.pt)] [[Model (650M)](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt)] [[Model (3B)](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt)] [[Model (15B)](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t48_15B_UR50D.pt)]

**(Ankh)** _Ankh: Optimized Protein Language Model Unlocks General-Purpose Modelling_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2301.06568)] [[GitHub](https://github.com/agemagician/Ankh)] [[Model (450M)](https://huggingface.co/ElnaggarLab/ankh-base)] [[Model (1.1B)](https://huggingface.co/ElnaggarLab/ankh-large)]

**(DNABERT-2)** _DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2306.15006)] [[GitHub](https://github.com/Zhihan1996/DNABERT_2)] [[Model (Base)](https://huggingface.co/zhihan1996/DNABERT-2-117M)]

**(xTrimoPGLM)** _xTrimoPGLM: Unified 100B-Scale Pre-trained Transformer for Deciphering the Language of Protein_ ```bioRxiv 2023```     
[[Paper](https://www.biorxiv.org/content/10.1101/2023.07.05.547496)]

