# Awesome Scientific Language Models
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
[![Stars](https://img.shields.io/github/stars/yuzhimanhua/Awesome-Scientific-Language-Models?style=social)](https://img.shields.io/github/stars/yuzhimanhua/Awesome-Scientific-Language-Models?style=social)

[![Papers](https://img.shields.io/badge/PaperNumber-211-blue)](https://img.shields.io/badge/PaperNumber-211-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRWelcome](https://img.shields.io/badge/PRs-Welcome-red)](https://img.shields.io/badge/PRs-Welcome-red)

A curated list of pre-trained language models in scientific domains (e.g., **mathematics**, **physics**, **chemistry**, **biology**, **medicine**, **materials science**, and **geoscience**), covering different model sizes (from **<100M** to **70B parameters**) and modalities (e.g., **language**, **vision**, **graph**, **molecule**, **protein**, **genome**, and **climate time series**). The repository will be continuously updated.

**NOTE 1**: To avoid ambiguity, when we talk about the number of parameters in a model, "Base" refers to 110M (i.e., BERT-Base), and "Large" refers to 340M (i.e., BERT-Large). Other numbers will be written explicitly.

**NOTE 2**: In each subsection, papers are sorted chronologically. If a paper has a preprint (e.g., arXiv or bioRxiv) version, its publication date is according to the preprint service. Otherwise, its publication date is according to the conference proceeding or journal.

**NOTE 3**: We appreciate contributions. If you have any suggested papers, feel free to reach out to yuz9@illinois.edu or submit a [pull request](https://github.com/yuzhimanhua/Awesome-Scientific-Language-Models/pulls). For format consistency, we will include a paper after (1) it has a version with author names AND (2) its GitHub and/or Hugging Face links are available.





## Contents
- [General](#general)
  - [Language](#general-language)
  - [Language + Graph](#general-language-graph)
- [Mathematics](#mathematics)
  - [Language](#mathematics-language)
  - [Language + Vision](#mathematics-language-vision)
  - [Other Modalities (Table)](#mathematics-other-modalities-table)
- [Physics](#physics)
  - [Language](#physics-language)
- [Chemistry and Materials Science](#chemistry-and-materials-science)
  - [Language](#chemistry-language)
  - [Language + Graph](#chemistry-language-graph-other-modalities-molecule)
  - [Language + Vision](#chemistry-language-vision)
  - [Other Modalities (Molecule)](#chemistry-language-graph-other-modalities-molecule)
- [Biology and Medicine](#biology-and-medicine)
  - [Language](#biology-language)
  - [Language + Graph](#biology-language-graph)
  - [Language + Vision](#biology-language-vision)
  - [Other Modalities (Protein)](#biology-other-modalities-protein)
  - [Other Modalities (DNA)](#biology-other-modalities-dna)
  - [Other Modalities (RNA)](#biology-other-modalities-rna)
  - [Other Modalities (Multiomics)](#biology-other-modalities-multiomics)
- [Geography, Geology, and Environmental Science](#geography-geology-and-environmental-science)
  - [Language](#geography-language)
  - [Language + Graph](#geography-language-graph)
  - [Language + Vision](#geography-language-vision)
  - [Other Modalities (Climate Time Series)](#geography-other-modalities-climate-time-series)





## General
<h3 id="general-language">Language</h3>

- **(SciBERT)** _SciBERT: A Pretrained Language Model for Scientific Text_ ```EMNLP 2019```     
[[Paper](https://arxiv.org/abs/1903.10676)] [[GitHub](https://github.com/allenai/scibert)] [[Model (Base)](https://huggingface.co/allenai/scibert_scivocab_uncased)]

- **(SciGPT2)** _Explaining Relationships between Scientific Documents_ ```ACL 2021```     
[[Paper](https://arxiv.org/abs/2002.00317)] [[GitHub](https://github.com/Kel-Lu/SciGen)] [[Model (117M)](https://drive.google.com/file/d/1AoNYnhvI6tensnrpQVc09KL1NWJ5MvFU/view)]

- **(CATTS)** _TLDR: Extreme Summarization of Scientific Documents_ ```EMNLP 2020 Findings```     
[[Paper](https://arxiv.org/abs/2004.15011)] [[GitHub](https://github.com/allenai/scitldr)] [[Model (406M)](https://storage.cloud.google.com/skiff-models/scitldr/catts-xsum.tldr-aic.pt)]

- **(SciNewsBERT)** _SciClops: Detecting and Contextualizing Scientific Claims for Assisting Manual Fact-Checking_ ```CIKM 2021```     
[[Paper](https://arxiv.org/abs/2110.13090)] [[Model (Base)](https://huggingface.co/psmeros/SciNewsBERT)]

- **(ScholarBERT)** _The Diminishing Returns of Masked Language Models to Science_ ```ACL 2023 Findings```     
[[Paper](https://arxiv.org/abs/2205.11342)] [[Model (Large)](https://huggingface.co/globuslabs/ScholarBERT)] [[Model (770M)](https://huggingface.co/globuslabs/ScholarBERT-XL)]

- **(AcademicRoBERTa)** _A Japanese Masked Language Model for Academic Domain_ ```COLING 2022 Workshop```     
[[Paper](https://aclanthology.org/2022.sdp-1.16)] [[GitHub](https://github.com/EhimeNLP/AcademicRoBERTa)] [[Model (125M)](https://huggingface.co/EhimeNLP/AcademicRoBERTa)]

- **(Galactica)** _Galactica: A Large Language Model for Science_ ```arXiv 2022```     
[[Paper](https://arxiv.org/abs/2211.09085)]

- **(DARWIN)** _DARWIN Series: Domain Specific Large Language Models for Natural Science_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2308.13565)] [[GitHub](https://github.com/MasterAI-EAM/Darwin)] [[Model (7B)](https://aigreendynamics-my.sharepoint.com/:f:/g/personal/yuwei_greendynamics_com_au/Euu1OzZTOS5OsQvVTRNV_gcBa67ehvk6uN6hJIHnBLOkDg?e=x5wxfk)]

- **(FORGE)** _FORGE: Pre-Training Open Foundation Models for Science_ ```SC 2023```     
[[Paper](https://doi.org/10.1145/3581784.3613215)] [[GitHub](https://github.com/at-aaims/forge)] [[Model (1.4B, General)](https://www.dropbox.com/sh/byr1ydik5n1ucod/AADOu_9C6AwVPTThTUFQ7yQba?dl=0)] [[Model (1.4B, Biology/Medicine)](https://www.dropbox.com/sh/41sqapgza3ok9q9/AADLgwTiHVU26ZeW_UQ8apyta?dl=0)] [[Model (1.4B, Chemistry)](https://www.dropbox.com/sh/1jn3n7099r8pzt8/AAAO6sOpFYG-G_qFI6C6CXVVa?dl=0)] [[Model (1.4B, Engineering)](https://www.dropbox.com/sh/ueki0n6y3v8gtkw/AAB6-3ml9slcbOonk6ccdD4Ua?dl=0)] [[Model (1.4B, Materials Science)](https://www.dropbox.com/sh/ngrr3bjulc76944/AABpm_OxA-GQPWzIPM4KpVKOa?dl=0)] [[Model (1.4B, Physics)](https://www.dropbox.com/sh/jxux4tplw5aw7kw/AAAdk334IEMbY7HJlJrWVzyfa?dl=0)] [[Model (1.4B, Social Science/Art)](https://www.dropbox.com/sh/54tuyslytqhpq1z/AAAc65c3TQWo2MyPoSiPxKI2a?dl=0)] [[Model (13B, General)](https://www.dropbox.com/sh/g53ot3dpqfsf6fr/AAB_RFeox2tbDKVFCH0QCw5pa?dl=0)] [[Model (22B, General)](https://www.dropbox.com/sh/7b9gbgcqdyph8v9/AABjNTaYu5PTjTMLb4-t6-PNa?dl=0)]

- **(SciGLM)** _SciGLM: Training Scientific Language Models with Self-Reflective Instruction Annotation and Tuning_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2401.07950)] [[GitHub](https://github.com/THUDM/SciGLM)]


<h3 id="general-language-graph">Language + Graph</h3>

- **(SPECTER)** _SPECTER: Document-level Representation Learning using Citation-informed Transformers_ ```ACL 2020```     
[[Paper](https://arxiv.org/abs/2004.07180)] [[GitHub](https://github.com/allenai/specter)] [[Model (Base)](https://huggingface.co/allenai/specter)]

- **(OAG-BERT)** _OAG-BERT: Towards a Unified Backbone Language Model for Academic Knowledge Services_ ```KDD 2022```     
[[Paper](https://arxiv.org/abs/2103.02410)] [[GitHub](https://github.com/THUDM/OAG-BERT)]

- **(ASPIRE)** _Multi-Vector Models with Textual Guidance for Fine-Grained Scientific Document Similarity_ ```NAACL 2022```     
[[Paper](https://arxiv.org/abs/2111.08366)] [[GitHub](https://github.com/allenai/aspire)] [[Model (Base)](https://huggingface.co/allenai/aspire-sentence-embedder)]

- **(SciNCL)** _Neighborhood Contrastive Learning for Scientific Document Representations with Citation Embeddings_ ```EMNLP 2022```     
[[Paper](https://arxiv.org/abs/2202.06671)] [[GitHub](https://github.com/malteos/scincl)] [[Model (Base)](https://huggingface.co/malteos/scincl)]

- **(SPECTER 2.0)** _SciRepEval: A Multi-Format Benchmark for Scientific Document Representations_ ```EMNLP 2023```     
[[Paper](https://arxiv.org/abs/2211.13308)] [[GitHub](https://github.com/allenai/SPECTER2)] [[Model (113M)](https://huggingface.co/allenai/specter2)]

- **(SciPatton)** _Patton: Language Model Pretraining on Text-Rich Networks_ ```ACL 2023```     
[[Paper](https://arxiv.org/abs/2305.12268)] [[GitHub](https://github.com/PeterGriffinJin/Patton)]

- **(SciMult)** _Pre-training Multi-task Contrastive Learning Models for Scientific Literature Understanding_ ```EMNLP 2023 Findings```     
[[Paper](https://arxiv.org/abs/2305.14232)] [[GitHub](https://github.com/yuzhimanhua/SciMult)] [[Model (138M)](https://huggingface.co/yuz9yuz/SciMult)]





## Mathematics
<h3 id="mathematics-language">Language</h3>

- **(GenBERT)** _Injecting Numerical Reasoning Skills into Language Models_ ```ACL 2020```     
[[Paper](https://arxiv.org/abs/2004.04487)] [[GitHub](https://github.com/ag1988/injecting_numeracy)]

- **(EPT)** _Point to the Expression: Solving Algebraic Word Problems using the Expression-Pointer Transformer Model_ ```NeurIPS 2020```     
[[Paper](https://aclanthology.org/2020.emnlp-main.308)] [[GitHub](https://github.com/snucclab/EPT)]

- **(MathBERT)** _MathBERT: A Pre-trained Language Model for General NLP Tasks in Mathematics Education_ ```arXiv 2021```     
[[Paper](https://arxiv.org/abs/2106.07340)] [[GitHub](https://github.com/tbs17/MathBERT)] [[Model (Base)](https://huggingface.co/tbs17/MathBERT)]

- **(MWP-BERT)** _MWP-BERT: Numeracy-Augmented Pre-training for Math Word Problem Solving_ ```NAACL 2022 Findings```     
[[Paper](https://arxiv.org/abs/2107.13435)] [[GitHub](https://github.com/LZhenwen/MWP-BERT)] [[Model (Base)](https://drive.google.com/drive/folders/1QC7b6dnUSbHLJQHJQNwecPNiQQoBFu8T)]

- **(BERT-TD)** _Seeking Patterns, Not just Memorizing Procedures: Contrastive Learning for Solving Math Word Problems_ ```ACL 2022 Findings```     
[[Paper](https://arxiv.org/abs/2110.08464)] [[GitHub](https://github.com/zwx980624/mwp-cl)]

- **(GSM8K-GPT)** _Training Verifiers to Solve Math Word Problems_ ```arXiv 2021```     
[[Paper](https://arxiv.org/abs/2110.14168)] [[GitHub](https://github.com/openai/grade-school-math)]

- **(DeductReasoner)** _Learning to Reason Deductively: Math Word Problem Solving as Complex Relation Extraction_ ```ACL 2022```     
[[Paper](https://arxiv.org/abs/2203.10316)] [[GitHub](https://github.com/allanj/Deductive-MWP)] [[Model (125M)](https://drive.google.com/file/d/1TAHbdCKar0gqFzOd76LIYMQyI6hPOmL0/view)]

- **(NaturalProver)** _NaturalProver: Grounded Mathematical Proof Generation with Language Models_ ```NeurIPS 2022```     
[[Paper](https://arxiv.org/abs/2205.12910)] [[GitHub](https://github.com/wellecks/naturalprover)]

- **(Minerva)** _Solving Quantitative Reasoning Problems with Language Models_ ```NeurIPS 2022```     
[[Paper](https://arxiv.org/abs/2206.14858)]

- **(Bhaskara)** _Lila: A Unified Benchmark for Mathematical Reasoning_ ```EMNLP 2022```     
[[Paper](https://arxiv.org/abs/2210.17517)] [[GitHub](https://github.com/allenai/Lila)] [[Model (2.7B)](https://huggingface.co/allenai/bhaskara)]

- **(WizardMath)** _WizardMath: Empowering Mathematical Reasoning for Large Language Models via Reinforced Evol-Instruct_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2308.09583)] [[GitHub](https://github.com/nlpxucan/WizardLM)] [[Model (7B)](https://huggingface.co/WizardLM/WizardMath-7B-V1.1)] [[Model (13B)](https://huggingface.co/WizardLM/WizardMath-13B-V1.0)] [[Model (70B)](https://huggingface.co/WizardLM/WizardMath-70B-V1.0)]

- **(MAmmoTH)** _MAmmoTH: Building Math Generalist Models through Hybrid Instruction Tuning_ ```ICLR 2024```     
[[Paper](https://arxiv.org/abs/2309.05653)] [[GitHub](https://github.com/TIGER-AI-Lab/MAmmoTH)] [[Model (7B)](https://huggingface.co/TIGER-Lab/MAmmoTH-7B)] [[Model (13B)](https://huggingface.co/TIGER-Lab/MAmmoTH-13B)] [[Model (70B)](https://huggingface.co/TIGER-Lab/MAmmoTH-70B)]

- **(MetaMath)** _MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models_ ```ICLR 2024```     
[[Paper](https://arxiv.org/abs/2309.12284)] [[GitHub](https://github.com/meta-math/MetaMath)] [[Model (7B)](https://huggingface.co/meta-math/MetaMath-7B-V1.0)] [[Model (13B)](https://huggingface.co/meta-math/MetaMath-13B-V1.0)] [[Model (70B)](https://huggingface.co/meta-math/MetaMath-70B-V1.0)]

- **(ToRA)** _ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving_ ```ICLR 2024```     
[[Paper](https://arxiv.org/abs/2309.17452)] [[GitHub](https://github.com/microsoft/ToRA)] [[Model (7B)](https://huggingface.co/llm-agents/tora-7b-v1.0)] [[Model (13B)](https://huggingface.co/llm-agents/tora-13b-v1.0)] [[Model (70B)](https://huggingface.co/llm-agents/tora-70b-v1.0)]

- **(Llemma)** _Llemma: An Open Language Model For Mathematics_ ```ICLR 2024```     
[[Paper](https://arxiv.org/abs/2310.10631)] [[GitHub](https://github.com/EleutherAI/math-lm)] [[Model (7B)](https://huggingface.co/EleutherAI/llemma_7b)] [[Model (34B)](https://huggingface.co/EleutherAI/llemma_34b)]

- **(OVM)** _Outcome-supervised Verifiers for Planning in Mathematical Reasoning_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2311.09724)] [[GitHub](https://github.com/FreedomIntelligence/OVM)] [[Model (7B)](https://huggingface.co/FreedomIntelligence/OVM-llama2-7b)]


<h3 id="mathematics-language-vision">Language + Vision</h3>

- **(Inter-GPS)** _Inter-GPS: Interpretable Geometry Problem Solving with Formal Language and Symbolic Reasoning_ ```ACL 2021```     
[[Paper](https://arxiv.org/abs/2105.04165)] [[GitHub](https://github.com/lupantech/InterGPS)]

- **(Geoformer)** _UniGeo: Unifying Geometry Logical Reasoning via Reformulating Mathematical Expression_ ```EMNLP 2022```     
[[Paper](https://arxiv.org/abs/2212.02746)] [[GitHub](https://github.com/chen-judge/UniGeo)]

- **(SCA-GPS)** _A Symbolic Character-Aware Model for Solving Geometry Problems_ ```ACM MM 2023```     
[[Paper](https://arxiv.org/abs/2308.02823)] [[GitHub](https://github.com/ning-mz/sca-gps)]

- **(UniMath-Flan-T5)** _UniMath: A Foundational and Multimodal Mathematical Reasoner_ ```EMNLP 2023```     
[[Paper](https://aclanthology.org/2023.emnlp-main.440)] [[GitHub](https://github.com/Zhenwen-NLP/UniMath)]

- **(G-LLaVA)** _G-LLaVA: Solving Geometric Problem with Multi-Modal Large Language Model_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2312.11370)] [[GitHub](https://github.com/pipilurj/G-LLaVA)]


<h3 id="mathematics-other-modalities-table">Other Modalities (Table)</h3>

- **(TAPAS)** _TAPAS: Weakly Supervised Table Parsing via Pre-training_ ```ACL 2020```     
[[Paper](https://arxiv.org/abs/2004.02349)] [[GitHub](https://github.com/google-research/tapas)] [[Model (Base)](https://storage.googleapis.com/tapas_models/2020_04_21/tapas_base.zip)] [[Model (Large)](https://storage.googleapis.com/tapas_models/2020_04_21/tapas_large.zip)]

- **(TaBERT)** _TaBERT: Learning Contextual Representations for Natural Language Utterances and Structured Tables_ ```ACL 2020```     
[[Paper](https://arxiv.org/abs/2005.08314)] [[GitHub](https://github.com/facebookresearch/TaBERT)] [[Model (Base)](https://drive.google.com/drive/folders/1fDW9rLssgDAv19OMcFGgFJ5iyd9p7flg)] [[Model (Large)](https://drive.google.com/drive/folders/1fDW9rLssgDAv19OMcFGgFJ5iyd9p7flg)]

- **(GraPPa)** _GraPPa: Grammar-Augmented Pre-Training for Table Semantic Parsing_ ```ICLR 2021```     
[[Paper](https://arxiv.org/abs/2009.13845)] [[GitHub](https://github.com/taoyds/grappa)] [[Model (355M)](https://huggingface.co/Salesforce/grappa_large_jnt)]

- **(TUTA)** _TUTA: Tree-based Transformers for Generally Structured Table Pre-training_ ```KDD 2021```     
[[Paper](https://arxiv.org/abs/2010.12537)] [[GitHub](https://github.com/microsoft/TUTA_table_understanding)]

- **(RCI)** _Capturing Row and Column Semantics in Transformer Based Question Answering over Tables_ ```NAACL 2021```     
[[Paper](https://arxiv.org/abs/2104.08303)] [[GitHub](https://github.com/IBM/row-column-intersection)] [[Model (12M)](https://huggingface.co/michaelrglass/albert-base-rci-wikisql-row)]

- **(TABBIE)** _TABBIE: Pretrained Representations of Tabular Data_ ```NAACL 2021```     
[[Paper](https://arxiv.org/abs/2105.02584)] [[GitHub](https://github.com/SFIG611/tabbie)]

- **(TAPEX)** _TAPEX: Table Pre-training via Learning a Neural SQL Executor_ ```ICLR 2022```     
[[Paper](https://arxiv.org/abs/2107.07653)] [[GitHub](https://github.com/microsoft/Table-Pretraining)] [[Model (140M)](https://huggingface.co/microsoft/tapex-base)] [[Model (406M)](https://huggingface.co/microsoft/tapex-large)]

- **(FORTAP)** _FORTAP: Using Formulas for Numerical-Reasoning-Aware Table Pretraining_ ```ACL 2022```     
[[Paper](https://arxiv.org/abs/2109.07323)] [[GitHub](https://github.com/microsoft/TUTA_table_understanding)]

- **(OmniTab)** _OmniTab: Pretraining with Natural and Synthetic Data for Few-shot Table-based Question Answering_ ```NAACL 2022```     
[[Paper](https://arxiv.org/abs/2207.03637)] [[GitHub](https://github.com/jzbjyb/OmniTab)] [[Model (406M)](https://huggingface.co/neulab/omnitab-large)]

- **(ReasTAP)** _ReasTAP: Injecting Table Reasoning Skills During Pre-training via Synthetic Reasoning Examples_ ```EMNLP 2022```     
[[Paper](https://arxiv.org/abs/2210.12374)] [[GitHub](https://github.com/Yale-LILY/ReasTAP)] [[Model (406M)](https://huggingface.co/Yale-LILY/reastap-large)]

- **(TableLlama)** _TableLlama: Towards Open Large Generalist Models for Tables_ ```NAACL 2024```     
[[Paper](https://arxiv.org/abs/2311.09206)] [[GitHub](https://github.com/OSU-NLP-Group/TableLlama)] [[Model (7B)](https://huggingface.co/osunlp/TableLlama)]





## Physics
<h3 id="physics-language">Language</h3>

- **(astroBERT)** _Building astroBERT, a Language Model for Astronomy & Astrophysics_ ```arXiv 2021```     
[[Paper](https://arxiv.org/abs/2112.00590)] [[Model (Base)](https://huggingface.co/adsabs/astroBERT)]

- **(AstroLLaMA)** _AstroLLaMA: Towards Specialized Foundation Models in Astronomy_ ```AACL 2023 Workshop```     
[[Paper](https://arxiv.org/abs/2309.06126)] [[Model (7B)](https://huggingface.co/universeTBD/astrollama)]

- **(AstroLLaMA-Chat)** _AstroLLaMA-Chat: Scaling AstroLLaMA with Conversational and Diverse Datasets_ ```Research Notes of the AAS 2024```     
[[Paper](https://arxiv.org/abs/2401.01916)] [[Model (7B)](https://huggingface.co/spaces/universeTBD/astrollama-7b-chat-alpha)]





## Chemistry and Materials Science
<h3 id="chemistry-language">Language</h3>

- **(ChemBERT)** _Automated Chemical Reaction Extraction from Scientific Literature_ ```Journal of Chemical Information and Modeling 2022```     
[[Paper](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00284)] [[GitHub](https://github.com/jiangfeng1124/ChemRxnExtractor)] [[Model (Base)](https://huggingface.co/jiangg/chembert_cased)]

- **(MatSciBERT)** _MatSciBERT: A Materials Domain Language Model for Text Mining and Information Extraction_ ```npj Computational Materials 2022```     
[[Paper](https://arxiv.org/abs/2109.15290)] [[GitHub](https://github.com/M3RG-IITD/MatSciBERT)] [[Model (Base)](https://huggingface.co/m3rg-iitd/matscibert)]

- **(MatBERT)** _Quantifying the Advantage of Domain-Specific Pre-training on Named Entity Recognition Tasks in Materials Science_ ```Patterns 2022```     
[[Paper](https://doi.org/10.1016/j.patter.2022.100488)] [[GitHub](https://github.com/lbnlp/MatBERT)]

- **(BatteryBERT)** _BatteryBERT: A Pretrained Language Model for Battery Database Enhancement_ ```Journal of Chemical Information and Modeling 2022```     
[[Paper](https://pubs.acs.org/doi/10.1021/acs.jcim.2c00035)] [[GitHub](https://github.com/ShuHuang/batterybert)] [[Model (Base)](https://huggingface.co/batterydata/batterybert-cased)]

- **(MaterialsBERT)** _A General-Purpose Material Property Data Extraction Pipeline from Large Polymer Corpora using Natural Language Processing_ ```npj Computational Materials 2023```     
[[Paper](https://arxiv.org/abs/2209.13136)] [[Model (Base)](https://huggingface.co/pranav-s/MaterialsBERT)]

- **(LLM-Prop)** _LLM-Prop: Predicting Physical and Electronic Properties of Crystalline Solids from Their Text Descriptions_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2310.14029)] [[GitHub](https://github.com/vertaix/LLM-Prop)]

- **(ChemDFM)** _ChemDFM: Dialogue Foundation Model for Chemistry_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2401.14818)] [[GitHub](https://github.com/OpenDFM/ChemDFM)] [[Model (13B)](https://huggingface.co/OpenDFM/ChemDFM-13B-v1.0)]

- **(CrystalLLM)** _Fine-Tuned Language Models Generate Stable Inorganic Materials as Text_ ```ICLR 2024```     
[[Paper](https://arxiv.org/abs/2402.04379)] [[GitHub](https://github.com/facebookresearch/crystal-llm)]

- **(ChemLLM)** _ChemLLM: A Chemical Large Language Model_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2402.06852)] [[Model (7B)](https://huggingface.co/AI4Chem/ChemLLM-7B-Chat)]

- **(LlaSMol)** _LlaSMol: Advancing Large Language Models for Chemistry with a Large-Scale, Comprehensive, High-Quality Instruction Tuning Dataset_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2402.09391)] [[GitHub](https://github.com/OSU-NLP-Group/LLM4Chem)] [[Model (7B)](https://huggingface.co/osunlp/LlaSMol-Mistral-7B)]


<h3 id="chemistry-language-vision">Language + Vision</h3>

- **(MolScribe)** _MolScribe: Robust Molecular Structure Recognition with Image-To-Graph Generation_ ```Journal of Chemical Information and Modeling 2023```     
[[Paper](https://arxiv.org/abs/2205.14311)] [[GitHub](https://github.com/thomas0809/MolScribe)] [[Model (88M)](https://huggingface.co/yujieq/MolScribe)]

- **(GIT-Mol)** _GIT-Mol: A Multi-modal Large Language Model for Molecular Science with Graph, Image, and Text_ ```Computers in Biology and Medicine 2024```     
[[Paper](https://arxiv.org/abs/2308.06911)] [[GitHub](https://github.com/ai-hpc-research-team/git-mol)]


<h3 id="chemistry-language-graph-other-modalities-molecule">Language + Graph / Other Modalities (Molecule)</h3>

- **(SMILES-BERT)** _SMILES-BERT: Large Scale Unsupervised Pre-Training for Molecular Property Prediction_ ```ACM BCB 2019```     
[[Paper](https://dl.acm.org/doi/abs/10.1145/3307339.3342186)] [[GitHub](https://github.com/uta-smile/SMILES-BERT)]

- **(MAT)** _Molecule Attention Transformer_ ```arXiv 2020```     
[[Paper](https://arxiv.org/abs/2002.08264)] [[GitHub](https://github.com/ardigen/MAT)] [[Model (125M)](https://drive.google.com/file/d/11-TZj8tlnD7ykQGliO9bCrySJNBnYD2k/view)]

- **(ChemBERTa)** _ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction_ ```arXiv 2020```     
[[Paper](https://arxiv.org/abs/2010.09885)] [[GitHub](https://github.com/seyonechithrananda/bert-loves-chemistry)] [[Model (125M)](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1)]

- **(MolBERT)** _Molecular Representation Learning with Language Models and Domain-Relevant Auxiliary Tasks_ ```arXiv 2020```     
[[Paper](https://arxiv.org/abs/2011.13230)] [[GitHub](https://github.com/BenevolentAI/MolBERT)] [[Model (Base)](https://ndownloader.figshare.com/files/25611290)]

- **(RXNFP)** _Mapping the Space of Chemical Reactions using Attention-Based Neural Networks_ ```Nature Machine Intelligence 2021```     
[[Paper](https://arxiv.org/abs/2012.06051)] [[GitHub](https://github.com/rxn4chemistry/rxnfp)] [[Model (Base)](https://github.com/rxn4chemistry/rxnfp/tree/master/rxnfp/models/transformers/bert_pretrained)]

- **(RXNMapper)** _Extraction of Organic Chemistry Grammar from Unsupervised Learning of Chemical Reactions_ ```Science Advances 2021```     
[[Paper](https://www.science.org/doi/full/10.1126/sciadv.abe4166)] [[GitHub](https://github.com/rxn4chemistry/rxnmapper)]

- **(MoLFormer)** _Large-Scale Chemical Language Representations Capture Molecular Structure and Properties_ ```Nature Machine Intelligence 2022```     
[[Paper](https://arxiv.org/abs/2106.09553)] [[GitHub](https://github.com/IBM/molformer)] [[Model (47M)](https://huggingface.co/katielink/MoLFormer-XL)]

- **(Chemformer)** _Chemformer: A Pre-trained Transformer for Computational Chemistry_ ```Machine Learning: Science and Technology 2022```     
[[Paper](https://chemrxiv.org/engage/chemrxiv/article-details/60ee8a3eb95bdd06d062074b)] [[GitHub](https://github.com/MolecularAI/Chemformer)] [[Model (45M)](https://az.app.box.com/s/7eci3nd9vy0xplqniitpk02rbg9q2zcq/folder/144881804954)] [[Model (230M)](https://az.app.box.com/s/7eci3nd9vy0xplqniitpk02rbg9q2zcq/folder/144881806154)]

- **(MolGPT)** _MolGPT: Molecular Generation using a Transformer-Decoder Model_ ```Journal of Chemical Information and Modeling 2022```     
[[Paper](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00600)] [[GitHub](https://github.com/devalab/molgpt)]

- **(Text2Mol)** _Text2Mol: Cross-Modal Molecule Retrieval with Natural Language Queries_ ```EMNLP 2021```     
[[Paper](https://aclanthology.org/2021.emnlp-main.47)] [[GitHub](https://github.com/cnedwards/text2mol)]

- **(KV-PLM)** _A Deep-learning System Bridging Molecule Structure and Biomedical Text with Comprehension Comparable to Human Professionals_ ```Nature Communications 2022```     
[[Paper](https://www.nature.com/articles/s41467-022-28494-3)] [[GitHub](https://github.com/thunlp/KV-PLM)] [[Model (Base)](https://drive.google.com/drive/folders/1xig3-3JG63kR-Xqj1b9wkPEdxtfD_4IX)]

- **(T5Chem)** _Unified Deep Learning Model for Multitask Reaction Predictions with Explanation_ ```Journal of Chemical Information and Modeling 2022```     
[[Paper](https://pubs.acs.org/doi/10.1021/acs.jcim.1c01467)] [[GitHub](https://github.com/HelloJocelynLu/t5chem)]

- **(MolT5)** _Translation between Molecules and Natural Language_ ```EMNLP 2022```     
[[Paper](https://arxiv.org/abs/2204.11817)] [[GitHub](https://github.com/blender-nlp/MolT5)] [[Model (77M)](https://huggingface.co/laituan245/molt5-small)] [[Model (250M)](https://huggingface.co/laituan245/molt5-base)] [[Model (800M)](https://huggingface.co/laituan245/molt5-large)]

- **(ChemGPT)** _Neural Scaling of Deep Chemical Models_ ```Nature Machine Intelligence 2023```     
[[Paper](https://chemrxiv.org/engage/chemrxiv/article-details/627bddd544bdd532395fb4b5)] [[Model (4.7M)](https://huggingface.co/ncfrey/ChemGPT-4.7M)] [[Model (19M)](https://huggingface.co/ncfrey/ChemGPT-19M)] [[Model (1.2B)](https://huggingface.co/ncfrey/ChemGPT-1.2B)]

- **(MoMu)** _A Molecular Multimodal Foundation Model Associating Molecule Graphs with Natural Language_ ```arXiv 2022```     
[[Paper](https://arxiv.org/abs/2209.05481)] [[GitHub](https://github.com/bingsu12/momu)]

- **(MFBERT)** _Large-Scale Distributed Training of Transformers for Chemical Fingerprinting_ ```Journal of Chemical Information and Modeling 2022```     
[[Paper](https://pubs.acs.org/doi/10.1021/acs.jcim.2c00715)] [[GitHub](https://github.com/GouldGroup/MFBERT)]

- **(SPMM)** _Bidirectional Generation of Structure and Properties Through a Single Molecular Foundation Model_ ```Nature Communications 2024```     
[[Paper](https://arxiv.org/abs/2211.10590)] [[GitHub](https://github.com/jinhojsk515/SPMM)]

- **(MoleculeSTM)** _Multi-modal Molecule Structure-text Model for Text-based Retrieval and Editing_ ```Nature Machine Intelligence 2023```     
[[Paper](https://arxiv.org/abs/2212.10789)] [[GitHub](https://github.com/chao1224/MoleculeSTM)]

- **(MolGen)** _Domain-Agnostic Molecular Generation with Self-feedback_ ```ICLR 2024```     
[[Paper](https://arxiv.org/abs/2301.11259)] [[GitHub](https://github.com/zjunlp/MolGen)] [[Model (7B)](https://huggingface.co/zjunlp/MolGen-7b)]

- **(Text+Chem T5)** _Unifying Molecular and Textual Representations via Multi-task Language Modelling_ ```ICML 2023```     
[[Paper](https://arxiv.org/abs/2301.12586)] [[GitHub](https://github.com/GT4SD/gt4sd-core)] [[Model (60M)](https://huggingface.co/GT4SD/multitask-text-and-chemistry-t5-small-augm)] [[Model (220M)](https://huggingface.co/GT4SD/multitask-text-and-chemistry-t5-base-augm)]

- **(CLAMP)** _Enhancing Activity Prediction Models in Drug Discovery with the Ability to Understand Human Language_ ```ICML 2023```     
[[Paper](https://arxiv.org/abs/2303.03363)] [[GitHub](https://github.com/ml-jku/clamp)]

- **(GIMLET)** _GIMLET: A Unified Graph-Text Model for Instruction-Based Molecule Zero-Shot Learning_ ```NeurIPS 2023```     
[[Paper](https://arxiv.org/abs/2306.13089)] [[GitHub](https://github.com/zhao-ht/GIMLET)] [[Model (60M)](https://huggingface.co/haitengzhao/gimlet)]

- **(MolFM)** _MolFM: A Multimodal Molecular Foundation Model_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2307.09484)] [[GitHub](https://github.com/PharMolix/OpenBioMed)]

- **(CatBERTa)** _Catalyst Property Prediction with CatBERTa: Unveiling Feature Exploration Strategies through Large Language Models_ ```ACS Catalysis 2023```     
[[Paper](https://arxiv.org/abs/2309.00563)] [[GitHub](https://github.com/hoon-ock/CatBERTa)]

- **(GPT-MolBERTa)** _GPT-MolBERTa: GPT Molecular Features Language Model for Molecular Property Prediction_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2310.03030)] [[GitHub](https://github.com/Suryanarayanan-Balaji/GPT-MolBERTa)] [[Model (82M, BERT)](https://drive.google.com/file/d/1b39vo6OZIa76VacZL50oT7mbO6Yue6l6/view)] [[Model (82M, RoBERTa)](https://drive.google.com/file/d/1_vKE6Rb9A7PU0PVTu3u1XlaqoPXsp9sJ/view)]

- **(MolCA)** _MolCA: Molecular Graph-Language Modeling with Cross-Modal Projector and Uni-Modal Adapter_ ```EMNLP 2023```     
[[Paper](https://arxiv.org/abs/2310.12798)] [[GitHub](https://github.com/acharkq/MolCA)]

- **(ActFound)** _A Foundation Model for Bioactivity Prediction using Pairwise Meta-learning_ ```bioRxiv 2023```     
[[Paper](https://www.biorxiv.org/content/10.1101/2023.10.30.564861)] [[GitHub](https://github.com/BFeng14/ActFound)]

- **(InstructMol)** _InstructMol: Multi-Modal Integration for Building a Versatile and Reliable Molecular Assistant in Drug Discovery_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2311.16208)] [[GitHub](https://github.com/IDEA-XL/InstructMol)]

- **(PolyNC)** _PolyNC: A Natural and Chemical Language Model for the Prediction of Unified Polymer Properties_ ```Chemical Science 2024```     
[[Paper](https://pubs.rsc.org/en/Content/ArticleLanding/2023/SC/D3SC05079C)] [[GitHub](https://github.com/HKQiu/Unified_ML4Polymers)] [[Model (220M)](https://huggingface.co/hkqiu/PolyNC)]

- **(3D-MoLM)** _Towards 3D Molecule-Text Interpretation in Language Models_ ```ICLR 2024```     
[[Paper](https://arxiv.org/abs/2401.13923)] [[GitHub](https://github.com/lsh0520/3D-MoLM)] [[Model (7B)](https://huggingface.co/Sihangli/3D-MoLM)]





## Biology and Medicine
**Acknowledgment: We referred to Wang et al.'s survey paper [_Pre-trained Language Models in Biomedical Domain: A
Systematic Survey_](https://arxiv.org/abs/2110.05006) when writing some parts of this section.**

<h3 id="biology-language">Language</h3>

- **(BioBERT)** _BioBERT: A Pre-trained Biomedical Language Representation Model for Biomedical Text Mining_ ```Bioinformatics 2020```     
[[Paper](https://arxiv.org/abs/1901.08746)] [[GitHub](https://github.com/dmis-lab/biobert)] [[Model (Base)](https://huggingface.co/dmis-lab/biobert-base-cased-v1.2)] [[Model (Large)](https://huggingface.co/dmis-lab/biobert-large-cased-v1.1)]

- **(BioELMo)** _Probing Biomedical Embeddings from Language Models_ ```NAACL 2019 Workshop```     
[[Paper](https://arxiv.org/abs/1904.02181)] [[GitHub](https://github.com/Andy-jqa/bioelmo)] [[Model (93M)](https://drive.google.com/file/d/1BQIuWGoZDVWppiz9Cst-ZqWd2mLiY2nc/view)]

- **(ClinicalBERT, Alsentzer et al.)** _Publicly Available Clinical BERT Embeddings_ ```NAACL 2019 Workshop```     
[[Paper](https://arxiv.org/abs/1904.03323)] [[GitHub](https://github.com/EmilyAlsentzer/clinicalBERT)] [[Model (Base)](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)]

- **(ClinicalBERT, Huang et al.)** _ClinicalBERT: Modeling Clinical Notes and Predicting Hospital Readmission_ ```arXiv 2019```     
[[Paper](https://arxiv.org/abs/1904.05342)] [[GitHub](https://github.com/kexinhuang12345/clinicalBERT)] [[Model (Base)](https://drive.google.com/file/d/1X3WrKLwwRAVOaAfKQ_tkTi46gsPfY5EB/edit)]

- **(BlueBERT, f.k.a. NCBI-BERT)** _Transfer Learning in Biomedical Natural Language Processing: An Evaluation of BERT and ELMo on Ten Benchmarking Datasets_ ```ACL 2019 Workshop```     
[[Paper](https://arxiv.org/abs/1906.05474)] [[GitHub](https://github.com/ncbi-nlp/bluebert)] [[Model (Base)](https://huggingface.co/bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12)] [[Model (Large)](https://huggingface.co/bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16)]

- **(BEHRT)** _BEHRT: Transformer for Electronic Health Records_ ```Scientific Reports 2020```     
[[Paper](https://arxiv.org/abs/1907.09538)] [[GitHub](https://github.com/deepmedicine/BEHRT)]

- **(EhrBERT)** _Fine-Tuning Bidirectional Encoder Representations from Transformers (BERT)–Based Models on Large-Scale Electronic Health Record Notes: An Empirical Study_ ```JMIR Medical Informatics 2019```     
[[Paper](https://medinform.jmir.org/2019/3/e14830)] [[GitHub](https://github.com/umassbento/ehrbert)]

- **(Clinical XLNet)** _Clinical XLNet: Modeling Sequential Clinical Notes and Predicting Prolonged Mechanical Ventilation_ ```EMNLP 2020 Workshop```     
[[Paper](https://arxiv.org/abs/1912.11975)] [[GitHub](https://github.com/lindvalllab/clinicalXLNet)]

- **(ouBioBERT)** _Pre-training Technique to Localize Medical BERT and Enhance Biomedical BERT_ ```arXiv 2020```     
[[Paper](https://arxiv.org/abs/2005.07202)] [[GitHub](https://github.com/sy-wada/blue_benchmark_with_transformers)] [[Model (Base)](https://huggingface.co/seiya/oubiobert-base-uncased)]

- **(COVID-Twitter-BERT)** _COVID-Twitter-BERT: A Natural Language Processing Model to Analyse COVID-19 Content on Twitter_ ```Frontiers in Artificial Intelligence 2023```     
[[Paper](https://arxiv.org/abs/2005.07503)] [[GitHub](https://github.com/digitalepidemiologylab/covid-twitter-bert)] [[Model (Large)](https://huggingface.co/digitalepidemiologylab/covid-twitter-bert-v2)]

- **(Med-BERT)** _Med-BERT: Pretrained Contextualized Embeddings on Large-Scale Structured Electronic Health Records for Disease Prediction_ ```npj Digital Medicine 2021```     
[[Paper](https://arxiv.org/abs/2005.12833)] [[GitHub](https://github.com/ZhiGroup/Med-BERT)]

- **(Bio-ELECTRA)** _On the Effectiveness of Small, Discriminatively Pre-trained Language Representation Models for Biomedical Text Mining_ ```EMNLP 2020 Workshop```     
[[Paper](https://www.biorxiv.org/content/10.1101/2020.05.20.107003)] [[GitHub](https://github.com/SciCrunch/bio_electra)] [[Model (Base)](https://zenodo.org/records/3971235)]

- **(BiomedBERT, f.k.a. PubMedBERT)** _Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing_ ```ACM Transactions on Computing for Healthcare 2021```     
[[Paper](https://arxiv.org/abs/2007.15779)] [[Model (Base)](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract)] [[Model (Large)](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract)]

- **(MCBERT)** _Conceptualized Representation Learning for Chinese Biomedical Text Mining_ ```arXiv 2020```     
[[Paper](https://arxiv.org/abs/2008.10813)] [[GitHub](https://github.com/alibaba-research/ChineseBLUE)] [[Model (Base)](https://drive.google.com/file/d/1ccXRvaeox5XCNP_aSk_ttLBY695Erlok/view)]

- **(BRLTM)** _Bidirectional Representation Learning from Transformers using Multimodal Electronic Health Record Data to Predict Depression_ ```JBHI 2021```     
[[Paper](https://arxiv.org/abs/2009.12656)] [[GitHub](https://github.com/lanyexiaosa/brltm)]

- **(BioRedditBERT)** _COMETA: A Corpus for Medical Entity Linking in the Social Media_ ```EMNLP 2020```     
[[Paper](https://arxiv.org/abs/2010.03295)] [[GitHub](https://github.com/cambridgeltl/cometa)] [[Model (Base)](https://huggingface.co/cambridgeltl/BioRedditBERT-uncased)]

- **(BioMegatron)** _BioMegatron: Larger Biomedical Domain Language Model_ ```EMNLP 2020```     
[[Paper](https://arxiv.org/abs/2010.06060)] [[GitHub](https://github.com/NVIDIA/NeMo)] [[Model (345M)](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/biomegatron345m_biovocab_50k_uncased)]

- **(SapBERT)** _Self-Alignment Pretraining for Biomedical Entity Representations_ ```NAACL 2021```     
[[Paper](https://arxiv.org/abs/2010.11784)] [[GitHub](https://github.com/cambridgeltl/sapbert)] [[Model (Base)](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext)]

- **(ClinicalTransformer)** _Clinical Concept Extraction using Transformers_ ```JAMIA 2020```     
[[Paper](https://academic.oup.com/jamia/article-abstract/27/12/1935/5943218)] [[GitHub](https://github.com/uf-hobi-informatics-lab/ClinicalTransformerNER)] [[Model (Base, BERT)](https://transformer-models.s3.amazonaws.com/mimiciii_bert_10e_128b.zip)] [[Model (125M, RoBERTa)](https://transformer-models.s3.amazonaws.com/mimiciii_roberta_10e_128b.zip)] [[Model (12M, ALBERT)](https://transformer-models.s3.amazonaws.com/mimiciii_albert_10e_128b.zip)] [[Model (Base, ELECTRA)](https://transformer-models.s3.amazonaws.com/mimiciii_electra_5e_128b.zip)] [[Model (117M, XLNet)](https://transformer-models.s3.amazonaws.com/mimiciii_xlnet_5e_128b.zip)] [[Model (149M, Longformer)](https://transformer-models.s3.amazonaws.com/mimiciii_longformer_5e_128b.zip)] [[Model (139M, DeBERTa)](https://transformer-models.s3.amazonaws.com/mimiciii_deberta_10e_128b.tar.gz)]

- **(BioRoBERTa)** _Pretrained Language Models for Biomedical and Clinical Tasks: Understanding and Extending the State-of-the-Art_ ```EMNLP 2020 Workshop```     
[[Paper](https://aclanthology.org/2020.clinicalnlp-1.17)] [[GitHub](https://github.com/facebookresearch/bio-lm)] [[Model (125M)](https://dl.fbaipublicfiles.com/biolm/RoBERTa-base-PM-M3-Voc-train-longer-hf.tar.gz)] [[Model (355M)](https://dl.fbaipublicfiles.com/biolm/RoBERTa-large-PM-M3-Voc-hf.tar.gz)]

- **(RAD-BERT)** _Highly Accurate Classification of Chest Radiographic Reports using a Deep Learning Natural Language Model Pre-trained on 3.8 Million Text Reports_ ```Bioinformatics 2020```     
[[Paper](https://academic.oup.com/bioinformatics/article/36/21/5255/5875602)] [[GitHub](https://github.com/rAIdiance/bert-for-radiology)]

- **(BioMedBERT)** _BioMedBERT: A Pre-trained Biomedical Language Model for QA and IR_ ```COLING 2020```     
[[Paper](https://aclanthology.org/2020.coling-main.59)] [[GitHub](https://github.com/BioMedBERT/biomedbert)]

- **(LBERT)** _LBERT: Lexically Aware Transformer-Based Bidirectional Encoder Representation Model for Learning Universal Bio-Entity Relations_ ```Bioinformatics 2021```     
[[Paper](https://academic.oup.com/bioinformatics/article/37/3/404/5893949)] [[GitHub](https://github.com/warikoone/LBERT)]

- **(ELECTRAMed)** _ELECTRAMed: A New Pre-trained Language Representation Model for Biomedical NLP_ ```arXiv 2021```     
[[Paper](https://arxiv.org/abs/2104.09585)] [[GitHub](https://github.com/gmpoli/electramed)] [[Model (Base)](https://huggingface.co/giacomomiolo/electramed_base_scivocab_1M)]

- **(SciFive)** _SciFive: A Text-to-Text Transformer Model for Biomedical Literature_ ```arXiv 2021```     
[[Paper](https://arxiv.org/abs/2106.03598)] [[GitHub](https://github.com/justinphan3110/SciFive)] [[Model (220M)](https://huggingface.co/razent/SciFive-base-Pubmed_PMC)] [[Model (770M)](https://huggingface.co/razent/SciFive-large-Pubmed_PMC)]

- **(BioALBERT)** _Benchmarking for Biomedical Natural Language Processing Tasks with a Domain Specific ALBERT_ ```BMC Bioinformatics 2022```     
[[Paper](https://arxiv.org/abs/2107.04374)] [[GitHub](https://github.com/usmaann/BioALBERT)] [[Model (12M)](https://drive.google.com/file/d/1SIBd_-GETHhMiZ7BgMdDPEUDjOjtN_bH/view)] [[Model (18M)](https://drive.google.com/file/d/16KRtHf8Meze2Hcc4vK_GUNhG-9LY6_6P/view)]

- **(Clinical-Longformer)** _Clinical-Longformer and Clinical-BigBird: Transformers for Long Clinical Sequences_ ```arXiv 2021```     
[[Paper](https://arxiv.org/abs/2201.11838)] [[GitHub](https://github.com/luoyuanlab/Clinical-Longformer)] [[Model (149M, Longformer)](https://huggingface.co/yikuan8/Clinical-Longformer)] [[Model (Base, BigBird)](https://huggingface.co/yikuan8/Clinical-BigBird)]

- **(BioBART)** _BioBART: Pretraining and Evaluation of A Biomedical Generative Language Model_ ```ACL 2022 Workshop```     
[[Paper](https://arxiv.org/abs/2204.03905)] [[GitHub](https://github.com/GanjinZero/BioBART)] [[Model (140M)](https://huggingface.co/GanjinZero/biobart-base)] [[Model (406M)](https://huggingface.co/GanjinZero/biobart-large)]

- **(BioGPT)** _BioGPT: Generative Pre-trained Transformer for Biomedical Text Generation and Mining_ ```Briefings in Bioinformatics 2022```     
[[Paper](https://arxiv.org/abs/2210.10341)] [[GitHub](https://github.com/microsoft/BioGPT)] [[Model (355M)](https://huggingface.co/microsoft/biogpt)] [[Model (1.5B)](https://huggingface.co/microsoft/BioGPT-Large)]

- **(Med-PaLM)** _Large Language Models Encode Clinical Knowledge_ ```Nature 2023```     
[[Paper](https://arxiv.org/abs/2212.13138)]

- **(ChatDoctor)** _ChatDoctor: A Medical Chat Model Fine-Tuned on a Large Language Model Meta-AI (LLaMA) using Medical Domain Knowledge_ ```Cureus 2023```     
[[Paper](https://arxiv.org/abs/2303.14070)] [[GitHub](https://github.com/Kent0n-Li/ChatDoctor)]

- **(DoctorGLM)** _DoctorGLM: Fine-tuning your Chinese Doctor is not a Herculean Task_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2304.01097)] [[GitHub](https://github.com/xionghonglin/DoctorGLM)]

- **(BenTsao, f.k.a. HuaTuo)** _HuaTuo: Tuning LLaMA Model with Chinese Medical Knowledge_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2304.06975)] [[GitHub](https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese)]

- **(MedAlpaca)** _MedAlpaca - An Open-Source Collection of Medical Conversational AI Models and Training Data_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2304.08247)] [[GitHub](https://github.com/kbressem/medAlpaca)] [[Model (7B)](https://huggingface.co/medalpaca/medalpaca-7b)] [[Model (13B)](https://huggingface.co/medalpaca/medalpaca-13b)]

- **(PMC-LLaMA)** _PMC-LLaMA: Towards Building Open-source Language Models for Medicine_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2304.14454)] [[GitHub](https://github.com/chaoyi-wu/PMC-LLaMA)] [[Model (7B)](https://huggingface.co/chaoyi-wu/PMC_LLAMA_7B)] [[Model (13B)](https://huggingface.co/axiong/PMC_LLaMA_13B)]

- **(Med-PaLM 2)** _Towards Expert-Level Medical Question Answering with Large Language Models_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2305.09617)]

- **(GatorTronGPT)** _A Study of Generative Large Language Model for Medical Research and Healthcare_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2305.13523)] [[GitHub](https://github.com/uf-hobi-informatics-lab/GatorTronGPT)] [[Model (345M)](https://huggingface.co/UFNLP/gatortronS)]

- **(HuatuoGPT)** _HuatuoGPT, towards Taming Language Model to Be a Doctor_ ```EMNLP 2023 Findings```     
[[Paper](https://arxiv.org/abs/2305.15075)] [[GitHub](https://github.com/FreedomIntelligence/HuatuoGPT)] [[Model (7B)](https://huggingface.co/FreedomIntelligence/HuatuoGPT-7B)] [[Model (13B)](https://huggingface.co/FreedomIntelligence/HuatuoGPT-13b-delta)]

- **(MedCPT)** _MedCPT: Contrastive Pre-trained Transformers with Large-scale PubMed Search Logs for Zero-shot Biomedical Information Retrieval_ ```Bioinformatics 2023```     
[[Paper](https://arxiv.org/abs/2307.00589)] [[GitHub](https://github.com/ncbi/MedCPT)] [[Model (Base)](https://huggingface.co/ncbi/MedCPT-Query-Encoder)]

- **(DISC-MedLLM)** _DISC-MedLLM: Bridging General Large Language Models and Real-World Medical Consultation_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2308.14346)] [[GitHub](https://github.com/FudanDISC/DISC-MedLLM)] [[Model (13B)](https://huggingface.co/Flmc/DISC-MedLLM)]

- **(DRG-LLaMA)** _DRG-LLaMA: Tuning LLaMA Model to Predict Diagnosis-related Group for Hospitalized Patients_ ```npj Digital Medicine 2024```     
[[Paper](https://arxiv.org/abs/2309.12625)] [[GitHub](https://github.com/hanyin88/DRG-LLaMA)]

- **(HuatuoGPT-II)** _HuatuoGPT-II, One-stage Training for Medical Adaption of LLMs_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2311.09774)] [[GitHub](https://github.com/FreedomIntelligence/HuatuoGPT-II)] [[Model (7B)](https://huggingface.co/FreedomIntelligence/HuatuoGPT2-7B)] [[Model (13B)](https://huggingface.co/FreedomIntelligence/HuatuoGPT2-13B)] [[Model (34B)](https://huggingface.co/FreedomIntelligence/HuatuoGPT2-34B)]

- **(MEDITRON)** _MEDITRON-70B: Scaling Medical Pretraining for Large Language Models_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2311.16079)] [[GitHub](https://github.com/epfLLM/megatron-LLM)] [[Model (7B)](https://huggingface.co/epfl-llm/meditron-7b)] [[Model (70B)](https://huggingface.co/epfl-llm/meditron-70b)]

- **(PLLaMa)** _PLLaMa: An Open-source Large Language Model for Plant Science_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2401.01600)] [[GitHub](https://github.com/Xianjun-Yang/PLLaMa)] [[Model (7B)](https://huggingface.co/Xianjun/PLLaMa-7b-base)] [[Model (13B)](https://huggingface.co/Xianjun/PLLaMa-13b-base)]

- **(BioMistral)** _BioMistral: A Collection of Open-Source Pretrained Large Language Models for Medical Domains_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2402.10373)] [[Model (7B)](https://huggingface.co/BioMistral/BioMistral-7B)]

- **(BioMedLM, f.k.a. PubMedGPT)** _BioMedLM: a Domain-Specific Large Language Model for Biomedical Text_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2403.18421)] [[GitHub](https://github.com/stanford-crfm/BioMedLM)] [[Model (2.7B)](https://huggingface.co/stanford-crfm/BioMedLM)]


<h3 id="biology-language-graph">Language + Graph</h3>

- **(G-BERT)** _Pre-training of Graph Augmented Transformers for Medication Recommendation_ ```IJCAI 2019```     
[[Paper](https://arxiv.org/abs/1906.00346)] [[GitHub](https://github.com/jshang123/G-Bert)]

- **(CODER)** _CODER: Knowledge Infused Cross-Lingual Medical Term Embedding for Term Normalization_ ```JBI 2022```     
[[Paper](https://arxiv.org/abs/2011.02947)] [[GitHub](https://github.com/GanjinZero/CODER)] [[Model (Base)](https://huggingface.co/GanjinZero/coder_eng)]

- **(KeBioLM)** _Improving Biomedical Pretrained Language Models with Knowledge_ ```NAACL 2021 Workshop```     
[[Paper](https://arxiv.org/abs/2104.10344)] [[GitHub](https://github.com/GanjinZero/KeBioLM)] [[Model (155M)](https://drive.google.com/file/d/1kMbTsc9rPpBc-6ezEHjMbQLljW3SUWG9/edit)]

- **(MoP)** _Mixture-of-Partitions: Infusing Large Biomedical Knowledge Graphs into BERT_ ```EMNLP 2021```     
[[Paper](https://arxiv.org/abs/2109.04810)] [[GitHub](https://github.com/cambridgeltl/mop)]

- **(BioLinkBERT)** _LinkBERT: Pretraining Language Models with Document Links_ ```ACL 2022```     
[[Paper](https://arxiv.org/abs/2203.15827)] [[GitHub](https://github.com/michiyasunaga/LinkBERT)] [[Model (Base)](https://huggingface.co/michiyasunaga/BioLinkBERT-base)] [[Model (Large)](https://huggingface.co/michiyasunaga/BioLinkBERT-large)]

- **(DRAGON)** _Deep Bidirectional Language-Knowledge Graph Pretraining_ ```NeurIPS 2022```     
[[Paper](https://arxiv.org/abs/2210.09338)] [[GitHub](https://github.com/michiyasunaga/dragon)] [[Model (360M)](https://nlp.stanford.edu/projects/myasu/DRAGON/models/biomed_model.pt)]


<h3 id="biology-language-vision">Language + Vision</h3>

- **(ConVIRT)** _Contrastive Learning of Medical Visual Representations from Paired Images and Text_ ```MLHC 2022```     
[[Paper](https://arxiv.org/abs/2010.00747)] [[GitHub](https://github.com/yuhaozhang/convirt)]

- **(MedViLL)** _Multi-modal Understanding and Generation for Medical Images and Text via Vision-Language Pre-Training_ ```JBHI 2022```     
[[Paper](https://arxiv.org/abs/2105.11333)] [[GitHub](https://github.com/SuperSupermoon/MedViLL)]

- **(GLoRIA)** _GLoRIA: A Multimodal Global-Local Representation Learning Framework for Label-efficient Medical Image Recognition_ ```ICCV 2021```     
[[Paper](https://ieeexplore.ieee.org/document/9710099)] [[GitHub](https://github.com/marshuang80/gloria)]

- **(LoVT)** _Joint Learning of Localized Representations from Medical Images and Reports_ ```ECCV 2022```     
[[Paper](https://arxiv.org/abs/2112.02889)] [[GitHub](https://github.com/philip-mueller/lovt)]

- **(CvT2DistilGPT2)** _Improving Chest X-Ray Report Generation by Leveraging Warm Starting_ ```Artificial Intelligence in Medicine 2023```     
[[Paper](https://arxiv.org/abs/2201.09405)] [[GitHub](https://github.com/aehrc/cvt2distilgpt2)]

- **(BioViL)** _Making the Most of Text Semantics to Improve Biomedical Vision-Language Processing_ ```ECCV 2022```     
[[Paper](https://arxiv.org/abs/2204.09817)] [[GitHub](https://github.com/microsoft/hi-ml/tree/main/hi-ml-multimodal)]

- **(LViT)** _LViT: Language meets Vision Transformer in Medical Image Segmentation_ ```TMI 2022```     
[[Paper](https://arxiv.org/abs/2206.14718)] [[GitHub](https://github.com/HUANGLIZI/LViT)]

- **(M3AE)** _Multi-Modal Masked Autoencoders for Medical Vision-and-Language Pre-Training_ ```MICCAI 2022```     
[[Paper](https://arxiv.org/abs/2209.07098)] [[GitHub](https://github.com/zhjohnchan/M3AE)]

- **(ARL)** _Align, Reason and Learn: Enhancing Medical Vision-and-Language Pre-training with Knowledge_ ```ACM MM 2022```     
[[Paper](https://arxiv.org/abs/2209.07118)] [[GitHub](https://github.com/zhjohnchan/ARL)]

- **(CheXzero)** _Expert-Level Detection of Pathologies from Unannotated Chest X-ray Images via Self-Supervised Learning_ ```Nature Biomedical Engineering 2022```     
[[Paper](https://www.nature.com/articles/s41551-022-00936-9)] [[GitHub](https://github.com/rajpurkarlab/CheXzero)]

- **(MGCA)** _Multi-Granularity Cross-modal Alignment for Generalized Medical Visual Representation Learning_ ```NeurIPS 2022```     
[[Paper](https://arxiv.org/abs/2210.06044)] [[GitHub](https://github.com/HKU-MedAI/MGCA)]

- **(MedCLIP)** _MedCLIP: Contrastive Learning from Unpaired Medical Images and Text_ ```EMNLP 2022```     
[[Paper](https://arxiv.org/abs/2210.10163)] [[GitHub](https://github.com/RyanWangZf/MedCLIP)]

- **(BioViL-T)** _Learning to Exploit Temporal Structure for Biomedical Vision-Language Processing_ ```CVPR 2023```     
[[Paper](https://arxiv.org/abs/2301.04558)] [[GitHub](https://github.com/microsoft/hi-ml/tree/main/hi-ml-multimodal)] [[Model](https://huggingface.co/microsoft/BiomedVLP-BioViL-T)]

- **(BiomedCLIP)** _BiomedCLIP: A Multimodal Biomedical Foundation Model Pretrained from Fifteen Million Scientific Image-Text Pairs_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2303.00915)] [[Model](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)]

- **(RGRG)** _Interactive and Explainable Region-guided Radiology Report Generation_ ```CVPR 2023```     
[[Paper](https://arxiv.org/abs/2304.08295)] [[GitHub](https://github.com/ttanida/rgrg)]

- **(LLaVA-Med)** _LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day_ ```NeurIPS 2023```     
[[Paper](https://arxiv.org/abs/2306.00890)] [[GitHub](https://github.com/microsoft/LLaVA-Med)]

- **(MONET)** _Transparent Medical Image AI via an Image–Text Foundation Model Grounded in Medical Literature_ ```Nature Medicine 2024```     
[[Paper](https://www.medrxiv.org/content/10.1101/2023.06.07.23291119)] [[GitHub](https://github.com/suinleelab/MONET)]

- **(Med-PaLM M)** _Towards Generalist Biomedical AI_ ```NEJM AI 2024```     
[[Paper](https://arxiv.org/abs/2307.14334)] [[GitHub](https://github.com/kyegomez/Med-PaLM)]

- **(BioCLIP)** _BioCLIP: A Vision Foundation Model for the Tree of Life_ ```arXiv 2023```    
[[Paper](https://arxiv.org/abs/2311.18803)] [[Github](https://github.com/Imageomics/BioCLIP)] [[Model](https://huggingface.co/imageomics/bioclip)]


<h3 id="biology-other-modalities-protein">Other Modalities (Protein)</h3>

- **(ProtTrans)** _ProtTrans: Towards Cracking the Language of Life's Code Through Self-Supervised Deep Learning and High Performance Computing_ ```TPAMI 2021```     
[[Paper](https://arxiv.org/abs/2007.06225)] [[GitHub](https://github.com/agemagician/ProtTrans)] [[Model (Base, BERT)](https://huggingface.co/Rostlab/prot_bert)] [[Model (12M, ALBERT)](https://huggingface.co/Rostlab/prot_albert)] [[Model (117M, XLNet)](https://huggingface.co/Rostlab/prot_xlnet)] [[Model (3B, T5)](https://huggingface.co/Rostlab/prot_t5_xl_uniref50)] [[Model (11B, T5)](https://huggingface.co/Rostlab/prot_t5_xxl_uniref50)]

- **(ESM-1b)** _Biological Structure and Function Emerge from Scaling Unsupervised Learning to 250 Million Protein Sequences_ ```PNAS 2021```     
[[Paper](https://www.pnas.org/doi/10.1073/pnas.2016239118)] [[GitHub](https://github.com/facebookresearch/esm)] [[Model (650M)](https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt)]

- **(ESM-1v)** _Language Models Enable Zero-Shot Prediction of the Effects of Mutations on Protein Function_ ```NeurIPS 2021```     
[[Paper](https://www.biorxiv.org/content/10.1101/2021.07.09.450648)] [[GitHub](https://github.com/facebookresearch/esm)] [[Model (650M)](https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_1.pt)]

- **(ProteinBERT)** _ProteinBERT: A Universal Deep-Learning Model of Protein Sequence and Function_ ```Bioinformatics 2022```     
[[Paper](https://academic.oup.com/bioinformatics/article/38/8/2102/6502274)] [[GitHub](https://github.com/nadavbra/protein_bert)] [[Model (16M)](https://huggingface.co/GrimSqueaker/proteinBERT)]

- **(ProtGPT2)** _ProtGPT2 is a Deep Unsupervised Language Model for Protein Design_ ```Nature Communications 2022```     
[[Paper](https://www.biorxiv.org/content/10.1101/2022.03.09.483666)] [[Model (738M)](https://huggingface.co/nferruz/ProtGPT2)]

- **(ESM-IF1)** _Learning Inverse Folding from Millions of Predicted Structures_ ```ICML 2022```     
[[Paper](https://www.biorxiv.org/content/10.1101/2022.04.10.487779)] [[GitHub](https://github.com/facebookresearch/esm)] [[Model (124M)](https://dl.fbaipublicfiles.com/fair-esm/models/esm_if1_gvp4_t16_142M_UR50.pt)]

- **(ProGen)** _Large Language Models Generate Functional Protein Sequences across Diverse Families_ ```Nature Biotechnology 2023```     
[[Paper](https://www.nature.com/articles/s41587-022-01618-2)]

- **(ProGen2)** _ProGen2: Exploring the Boundaries of Protein Language Models_ ```Cell Systems 2023```     
[[Paper](https://arxiv.org/abs/2206.13517)] [[GitHub](https://github.com/salesforce/progen)] [[Model (151M)](https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-small.tar.gz)] [[Model (764M)](https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-base.tar.gz)] [[Model (2.7B)](https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-large.tar.gz)] [[Model (6.4B)](https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-xlarge.tar.gz)]

- **(ESM-2)** _Evolutionary-Scale Prediction of Atomic-Level Protein Structure with a Language Model_ ```Science 2023```     
[[Paper](https://www.biorxiv.org/content/10.1101/2022.07.20.500902)] [[GitHub](https://github.com/facebookresearch/esm)] [[Model (8M)](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t6_8M_UR50D.pt)] [[Model (35M)](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t12_35M_UR50D.pt)] [[Model (150M)](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t30_150M_UR50D.pt)] [[Model (650M)](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt)] [[Model (3B)](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt)] [[Model (15B)](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t48_15B_UR50D.pt)]

- **(Ankh)** _Ankh: Optimized Protein Language Model Unlocks General-Purpose Modelling_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2301.06568)] [[GitHub](https://github.com/agemagician/Ankh)] [[Model (450M)](https://huggingface.co/ElnaggarLab/ankh-base)] [[Model (1.1B)](https://huggingface.co/ElnaggarLab/ankh-large)]

- **(ProtST)** _ProtST: Multi-Modality Learning of Protein Sequences and Biomedical Texts_ ```ICML 2023```     
[[Paper](https://arxiv.org/abs/2301.12040)] [[GitHub](https://github.com/DeepGraphLearning/ProtST)]

- **(LM-Design)** _Structure-informed Language Models Are Protein Designers_ ```ICML 2023```     
[[Paper](https://arxiv.org/abs/2302.01649)] [[GitHub](https://github.com/BytedProtein/ByProt)] [[Model (650M)](https://zenodo.org/records/10046338/files/lm_design_esm2_650m.zip)]

- **(xTrimoPGLM)** _xTrimoPGLM: Unified 100B-Scale Pre-trained Transformer for Deciphering the Language of Protein_ ```bioRxiv 2023```     
[[Paper](https://www.biorxiv.org/content/10.1101/2023.07.05.547496)]

- **(Prot2Text)** _Prot2Text: Multimodal Protein's Function Generation with GNNs and Transformers_ ```AAAI 2024```     
[[Paper](https://arxiv.org/abs/2307.14367)] [[GitHub](https://github.com/hadi-abdine/Prot2Text)] [[Model (256M)](https://1drv.ms/u/s!AhcBGHWGY2mukdh4KujJfOJ_PxvJog?e=C6x4E6)] [[Model (283M)](https://1drv.ms/u/s!AhcBGHWGY2mukdh1N1kfnmXBEar-Tw?e=fACWFt)] [[Model (398M)](https://1drv.ms/u/s!AhcBGHWGY2mukdh3yPy98rqWfYcTJA?e=ot1SX6)] [[Model (898M)](https://1drv.ms/u/s!AhcBGHWGY2mukdh2EL4iP_IoVKu1tg?e=PioL6B)]

- **(SaProt)** _SaProt: Protein Language Modeling with Structure-Aware Vocabulary_ ```ICLR 2024```     
[[Paper](https://www.biorxiv.org/content/10.1101/2023.10.01.560349)] [[GitHub](https://github.com/westlake-repl/SaProt)] [[Model (35M)](https://huggingface.co/westlake-repl/SaProt_35M_AF2)] [[Model (650M)](https://huggingface.co/westlake-repl/SaProt_650M_AF2)]


<h3 id="biology-other-modalities-dna">Other Modalities (DNA)</h3>

- **(DNABERT)** _DNABERT: Pre-trained Bidirectional Encoder Representations from Transformers Model for DNA-Language in Genome_ ```Bioinformatics 2021```     
[[Paper](https://www.biorxiv.org/content/10.1101/2020.09.17.301879)] [[GitHub](https://github.com/jerryji1993/DNABERT)] [[Model (Base)](https://drive.google.com/file/d/1BJjqb5Dl2lNMg2warsFQ0-Xvn1xxfFXC/view)]

- **(Enformer)** _Effective Gene Expression Prediction from Sequence by Integrating Long-Range Interactions_ ```Nature Methods 2021```     
[[Paper](https://www.nature.com/articles/s41592-021-01252-x)] [[GitHub](https://github.com/google-deepmind/deepmind-research/tree/master/enformer)] [[Model (249M)](https://huggingface.co/EleutherAI/enformer-preview)]

- **(GenSLMs)** _GenSLMs: Genome-Scale Language Models Reveal SARS-CoV-2 Evolutionary Dynamics_ ```The International Journal of High Performance Computing Applications 2023```     
[[Paper](https://www.biorxiv.org/content/10.1101/2022.10.10.511571)] [[GitHub](https://github.com/ramanathanlab/genslm)]

- **(Nucleotide Transformer)** _The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics_ ```bioRxiv 2023```     
[[Paper](https://www.biorxiv.org/content/10.1101/2023.01.11.523679)] [[GitHub](https://github.com/instadeepai/nucleotide-transformer)] [[Model (50M)](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species)] [[Model (100M)](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-100m-multi-species)] [[Model (250M)](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-250m-multi-species)] [[Model (500M)](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species)]

- **(GENA-LM)** _GENA-LM: A Family of Open-Source Foundational DNA Language Models for Long Sequences_ ```bioRxiv 2023```     
[[Paper](https://www.biorxiv.org/content/10.1101/2023.06.12.544594)] [[GitHub](https://github.com/AIRI-Institute/GENA_LM)] [[Model (Base, BERT)](https://huggingface.co/AIRI-Institute/gena-lm-bert-base-t2t)] [[Model (Large, BERT)](https://huggingface.co/AIRI-Institute/gena-lm-bert-large-t2t)] [[Model (Base, BigBird)](https://huggingface.co/AIRI-Institute/gena-lm-bigbird-base-t2t)]

- **(DNABERT-2)** _DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genome_ ```ICLR 2024```     
[[Paper](https://arxiv.org/abs/2306.15006)] [[GitHub](https://github.com/Zhihan1996/DNABERT_2)] [[Model (Base)](https://huggingface.co/zhihan1996/DNABERT-2-117M)]

- **(HyenaDNA)** _HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution_ ```NeurIPS 2023```     
[[Paper](https://arxiv.org/abs/2306.15794)] [[GitHub](https://github.com/HazyResearch/hyena-dna)] [[Model (0.4M)](https://huggingface.co/LongSafari/hyenadna-tiny-1k-seqlen-hf)] [[Model (3.3M)](https://huggingface.co/LongSafari/hyenadna-small-32k-seqlen-hf)] [[Model (6.6M)](https://huggingface.co/LongSafari/hyenadna-large-1m-seqlen-hf)]


<h3 id="biology-other-modalities-rna">Other Modalities (RNA)</h3>

- **(RNABERT)** _Informative RNA-base Embedding for Functional RNA Structural Alignment and Clustering by Deep Representation Learning_ ```NAR Genomics and Bioinformatics 2022```     
[[Paper](https://www.biorxiv.org/content/10.1101/2021.08.23.457433)] [[GitHub](https://github.com/mana438/RNABERT)]

- **(RNA-FM)** _Interpretable RNA Foundation Model from Unannotated Data for Highly Accurate RNA Structure and Function Predictions_ ```arXiv 2022```     
[[Paper](https://arxiv.org/abs/2204.00300)] [[GitHub](https://github.com/ml4bio/RNA-FM)]

- **(RNA-MSM)** _Multiple Sequence-Alignment-Based RNA Language Model and its Application to Structural Inference_ ```Nucleic Acids Research 2024```     
[[Paper](https://www.biorxiv.org/content/10.1101/2023.03.15.532863)] [[GitHub](https://github.com/yikunpku/RNA-MSM)]


<h3 id="biology-other-modalities-multiomics">Other Modalities (Multiomics)</h3>

- **(scBERT)** _scBERT as a Large-scale Pretrained Deep Language Model for Cell Type Annotation of Single-cell RNA-seq Data_ ```Nature Machine Intelligence 2022```     
[[Paper](https://www.biorxiv.org/content/10.1101/2021.12.05.471261)] [[GitHub](https://github.com/TencentAILabHealthcare/scBERT)]

- **(scGPT)** _scGPT: Towards Building a Foundation Model for Single-Cell Multi-omics Using Generative AI_ ```Nature Methods 2024```     
[[Paper](https://www.biorxiv.org/content/10.1101/2023.04.30.538439)] [[GitHub](https://github.com/bowang-lab/scGPT)]

- **(scFoundation)** _Large Scale Foundation Model on Single-cell Transcriptomics_ ```bioRxiv 2023```     
[[Paper](https://www.biorxiv.org/content/10.1101/2023.05.29.542705)] [[GitHub](https://github.com/biomap-research/scFoundation)] [[Model (100M)](https://hopebio2020-my.sharepoint.com/:f:/g/personal/dongsheng_biomap_com/Eh22AX78_AVDv6k6v4TZDikBXt33gaWXaz27U9b1SldgbA)]

- **(Geneformer)** _Transfer Learning Enables Predictions in Network Biology_ ```Nature 2023```     
[[Paper](https://www.nature.com/articles/s41586-023-06139-9)] [[Model (10M)](https://huggingface.co/ctheodoris/Geneformer/blob/main/pytorch_model.bin)] [[Model (40M)](https://huggingface.co/ctheodoris/Geneformer/blob/main/geneformer-12L-30M/pytorch_model.bin)]

- **(CellLM)** _Large-Scale Cell Representation Learning via Divide-and-Conquer Contrastive Learning_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2306.04371)] [[GitHub](https://github.com/PharMolix/OpenBioMed)]

- **(BioMedGPT)** _BioMedGPT: Open Multimodal Generative Pre-trained Transformer for BioMedicine_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2308.09442)] [[GitHub](https://github.com/PharMolix/OpenBioMed)] [[Model (7B)](https://huggingface.co/PharMolix/BioMedGPT-LM-7B)] [[Model (10B)](https://huggingface.co/PharMolix/BioMedGPT-10B)]

- **(CellPLM)** _CellPLM: Pre-training of Cell Language Model Beyond Single Cells_ ```ICLR 2024```     
[[Paper](https://www.biorxiv.org/content/10.1101/2023.10.03.560734)] [[GitHub](https://github.com/OmicsML/CellPLM)] [[Model (82M)](https://www.dropbox.com/scl/fo/i5rmxgtqzg7iykt2e9uqm/h?rlkey=o8hi0xads9ol07o48jdityzv1&dl=0)]

- **(xTrimoGene)** _xTrimoGene: An Efficient and Scalable Representation Learner for Single-Cell RNA-Seq Data_ ```NeurIPS 2023```     
[[Paper](https://arxiv.org/abs/2311.15156)] [[GitHub](https://github.com/biomap-research/scFoundation)]





## Geography, Geology, and Environmental Science
<h3 id="geography-language">Language</h3>

- **(ClimateBERT)** _ClimateBERT: A Pretrained Language Model for Climate-Related Text_ ```arXiv 2021```     
[[Paper](https://arxiv.org/abs/2110.12010)] [[GitHub](https://github.com/climatebert/language-model)] [[Model (82M)](https://huggingface.co/climatebert/distilroberta-base-climate-f)]

- **(SpaBERT)** _SpaBERT: A Pretrained Language Model from Geographic Data for Geo-Entity Representation_ ```EMNLP 2022 Findings```     
[[Paper](https://arxiv.org/abs/2210.12213)] [[GitHub](https://github.com/zekun-li/spabert)] [[Model (Base)](https://drive.google.com/file/d/1l44FY3DtDxzM_YVh3RR6PJwKnl80IYWB/view)] [[Model (Large)](https://drive.google.com/file/d/1LeZayTR92R5bu9gH_cGCwef7nnMX35cR/view)]

- **(K2)** _K2: A Foundation Language Model for Geoscience Knowledge Understanding and Utilization_ ```WSDM 2024```     
[[Paper](https://arxiv.org/abs/2306.05064)] [[GitHub](https://github.com/davendw49/k2)] [[Model (7B)](https://huggingface.co/daven3/k2-v1)]

- **(OceanGPT)** _OceanGPT: A Large Language Model for Ocean Science Tasks_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2310.02031)] [[GitHub](https://github.com/zjunlp/KnowLM)] [[Model (7B)](https://huggingface.co/zjunlp/OceanGPT-7b)]

- **(ClimateBERT-NetZero)** _ClimateBERT-NetZero: Detecting and Assessing Net Zero and Reduction Targets_ ```EMNLP 2023```     
[[Paper](https://arxiv.org/abs/2310.08096)] [[Model (82M)](https://huggingface.co/climatebert/netzero-reduction)]

- **(GeoGalactica)** _GeoGalactica: A Scientific Large Language Model in Geoscience_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2401.00434)] [[GitHub](https://github.com/geobrain-ai/geogalactica)] [[Model (30B)](https://huggingface.co/geobrain-ai/geogalactica)]


<h3 id="geography-language-graph">Language + Graph</h3>

- **(ERNIE-GeoL)** _ERNIE-GeoL: A Geography-and-Language Pre-trained Model and its Applications in Baidu Maps_ ```KDD 2022```     
[[Paper](https://arxiv.org/abs/2203.09127)]

- **(PK-Chat)** _PK-Chat: Pointer Network Guided Knowledge Driven Generative Dialogue Model_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2304.00592)] [[GitHub](https://github.com/iiot-tbb/Dialogue_DDE)] [[Model (132M)](https://pan.baidu.com/s/1bNzGnnRPMfT4jkD_UsNSYg?pwd=pa7h)]


<h3 id="geography-language-vision">Language + Vision</h3>

- **(MGeo)** _MGeo: Multi-Modal Geographic Pre-Training Method_ ```SIGIR 2023```     
[[Paper](https://arxiv.org/abs/2301.04283)] [[GitHub](https://github.com/PhantomGrapes/MGeo)]


<h3 id="geography-other-modalities-climate-time-series">Other Modalities (Climate Time Series)</h3>

- **(Pangu-Weather)** _Accurate Medium-Range Global Weather Forecasting with 3D Neural Networks_ ```Nature 2023```     
[[Paper](https://arxiv.org/abs/2211.02556)] [[GitHub](https://github.com/198808xc/Pangu-Weather)]

- **(ClimaX)** _ClimaX: A Foundation Model for Weather and Climate_ ```ICML 2023```     
[[Paper](https://arxiv.org/abs/2301.10343)] [[GitHub](https://github.com/microsoft/ClimaX)]

- **(FengWu)** _FengWu: Pushing the Skillful Global Medium-Range Weather Forecast beyond 10 Days Lead_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2304.02948)] [[GitHub](https://github.com/OpenEarthLab/FengWu)]

- **(FuXi)** _FuXi: A Cascade Machine Learning Forecasting System for 15-day Global Weather Forecast_ ```npj Climate and Atmospheric Science 2023```     
[[Paper](https://arxiv.org/abs/2306.12873)] [[GitHub](https://github.com/tpys/FuXi)]
