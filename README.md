# Awesome Scientific Language Models
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
[![Stars](https://img.shields.io/github/stars/yuzhimanhua/Awesome-Scientific-Language-Models?style=social)](https://img.shields.io/github/stars/yuzhimanhua/Awesome-Scientific-Language-Models?style=social)

[![Papers](https://img.shields.io/badge/PaperNumber-264-blue)](https://img.shields.io/badge/PaperNumber-264-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRWelcome](https://img.shields.io/badge/PRs-Welcome-red)](https://img.shields.io/badge/PRs-Welcome-red)

A curated list of pre-trained language models in scientific domains (e.g., **mathematics**, **physics**, **chemistry**, **materials science**, **biology**, **medicine**, **geoscience**), covering different model sizes (from **100M** to **100B parameters**) and modalities (e.g., **language**, **graph**, **vision**, **table**, **molecule**, **protein**, **genome**, **climate time series**). 

The repository is part of our survey paper [**A Comprehensive Survey of Scientific Large Language Models and Their Applications in Scientific Discovery**](https://arxiv.org/abs/2406.10833) and will be continuously updated.

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
  - [Language + Graph](#chemistry-language-graph)
  - [Language + Vision](#chemistry-language-vision)
  - [Other Modalities (Molecule)](#chemistry-other-modalities-molecule)
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
[[Paper](https://arxiv.org/abs/2211.09085)] [[Model (125M)](https://huggingface.co/facebook/galactica-125m)] [[Model (1.3B)](https://huggingface.co/facebook/galactica-1.3b)] [[Model (6.7B)](https://huggingface.co/facebook/galactica-6.7b)] [[Model (30B)](https://huggingface.co/facebook/galactica-30b)] [[Model (120B)](https://huggingface.co/facebook/galactica-120b)]

- **(DARWIN)** _DARWIN Series: Domain Specific Large Language Models for Natural Science_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2308.13565)] [[GitHub](https://github.com/MasterAI-EAM/Darwin)] [[Model (7B)](https://aigreendynamics-my.sharepoint.com/:f:/g/personal/yuwei_greendynamics_com_au/Euu1OzZTOS5OsQvVTRNV_gcBa67ehvk6uN6hJIHnBLOkDg?e=x5wxfk)]

- **(FORGE)** _FORGE: Pre-training Open Foundation Models for Science_ ```SC 2023```     
[[Paper](https://doi.org/10.1145/3581784.3613215)] [[GitHub](https://github.com/at-aaims/forge)] [[Model (1.4B, General)](https://www.dropbox.com/sh/byr1ydik5n1ucod/AADOu_9C6AwVPTThTUFQ7yQba?dl=0)] [[Model (1.4B, Biology/Medicine)](https://www.dropbox.com/sh/41sqapgza3ok9q9/AADLgwTiHVU26ZeW_UQ8apyta?dl=0)] [[Model (1.4B, Chemistry)](https://www.dropbox.com/sh/1jn3n7099r8pzt8/AAAO6sOpFYG-G_qFI6C6CXVVa?dl=0)] [[Model (1.4B, Engineering)](https://www.dropbox.com/sh/ueki0n6y3v8gtkw/AAB6-3ml9slcbOonk6ccdD4Ua?dl=0)] [[Model (1.4B, Materials Science)](https://www.dropbox.com/sh/ngrr3bjulc76944/AABpm_OxA-GQPWzIPM4KpVKOa?dl=0)] [[Model (1.4B, Physics)](https://www.dropbox.com/sh/jxux4tplw5aw7kw/AAAdk334IEMbY7HJlJrWVzyfa?dl=0)] [[Model (1.4B, Social Science/Art)](https://www.dropbox.com/sh/54tuyslytqhpq1z/AAAc65c3TQWo2MyPoSiPxKI2a?dl=0)] [[Model (13B, General)](https://www.dropbox.com/sh/g53ot3dpqfsf6fr/AAB_RFeox2tbDKVFCH0QCw5pa?dl=0)] [[Model (22B, General)](https://www.dropbox.com/sh/7b9gbgcqdyph8v9/AABjNTaYu5PTjTMLb4-t6-PNa?dl=0)]

- **(SciGLM)** _SciGLM: Training Scientific Language Models with Self-Reflective Instruction Annotation and Tuning_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2401.07950)] [[GitHub](https://github.com/THUDM/SciGLM)] [[Model (6B)](https://huggingface.co/zd21/SciGLM-6B)]


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
[[Paper](https://arxiv.org/abs/2309.05653)] [[GitHub](https://github.com/TIGER-AI-Lab/MAmmoTH)] [[Model (7B, LLaMA-2)](https://huggingface.co/TIGER-Lab/MAmmoTH-7B)] [[Model (7B, Mistral)](https://huggingface.co/TIGER-Lab/MAmmoTH-7B-Mistral)] [[Model (13B, LLaMA-2)](https://huggingface.co/TIGER-Lab/MAmmoTH-13B)] [[Model (70B, LLaMA-2)](https://huggingface.co/TIGER-Lab/MAmmoTH-70B)]

- **(MetaMath)** _MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models_ ```ICLR 2024```     
[[Paper](https://arxiv.org/abs/2309.12284)] [[GitHub](https://github.com/meta-math/MetaMath)] [[Model (7B, LLaMA-2)](https://huggingface.co/meta-math/MetaMath-7B-V1.0)] [[Model (7B, Mistral)](https://huggingface.co/meta-math/MetaMath-Mistral-7B)] [[Model (13B, LLaMA-2)](https://huggingface.co/meta-math/MetaMath-13B-V1.0)] [[Model (70B, LLaMA-2)](https://huggingface.co/meta-math/MetaMath-70B-V1.0)]

- **(ToRA)** _ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving_ ```ICLR 2024```     
[[Paper](https://arxiv.org/abs/2309.17452)] [[GitHub](https://github.com/microsoft/ToRA)] [[Model (7B)](https://huggingface.co/llm-agents/tora-7b-v1.0)] [[Model (13B)](https://huggingface.co/llm-agents/tora-13b-v1.0)] [[Model (70B)](https://huggingface.co/llm-agents/tora-70b-v1.0)]

- **(MathCoder)** _MathCoder: Seamless Code Integration in LLMs for Enhanced Mathematical Reasoning_ ```ICLR 2024```     
[[Paper](https://arxiv.org/abs/2310.03731)] [[GitHub](https://github.com/mathllm/MathCoder)] [[Model (7B)](https://huggingface.co/MathLLM/MathCoder-L-7B)] [[Model (13B)](https://huggingface.co/MathLLM/MathCoder-L-13B)]

- **(Llemma)** _Llemma: An Open Language Model For Mathematics_ ```ICLR 2024```     
[[Paper](https://arxiv.org/abs/2310.10631)] [[GitHub](https://github.com/EleutherAI/math-lm)] [[Model (7B)](https://huggingface.co/EleutherAI/llemma_7b)] [[Model (34B)](https://huggingface.co/EleutherAI/llemma_34b)]

- **(OVM)** _OVM, Outcome-Supervised Value Models for Planning in Mathematical Reasoning_ ```NAACL 2024 Findings```     
[[Paper](https://arxiv.org/abs/2311.09724)] [[GitHub](https://github.com/FreedomIntelligence/OVM)] [[Model (7B, LLaMA-2)](https://huggingface.co/FreedomIntelligence/OVM-llama2-7b)] [[Model (7B, Mistral)](https://huggingface.co/FreedomIntelligence/OVM-Mistral-7b)]

- **(DeepSeekMath)** _DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2402.03300)] [[GitHub](https://github.com/deepseek-ai/DeepSeek-Math)] [[Model (7B)](https://huggingface.co/deepseek-ai/deepseek-math-7b-base)]

- **(InternLM-Math)** _InternLM-Math: Open Math Large Language Models Toward Verifiable Reasoning_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2402.06332)] [[GitHub](https://github.com/InternLM/InternLM-Math)] [[Model (7B)](https://huggingface.co/internlm/internlm2-math-base-7b)] [[Model (20B)](https://huggingface.co/internlm/internlm2-math-base-20b)]

- **(OpenMath)** _OpenMathInstruct-1: A 1.8 Million Math Instruction Tuning Dataset_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2402.10176)] [[Model (7B, Mistral)](https://huggingface.co/nvidia/OpenMath-Mistral-7B-v0.1-hf)] [[Model (70B, LLaMA-2)](https://huggingface.co/nvidia/OpenMath-Llama-2-70b-hf)]

- **(Rho-Math)** _Rho-1: Not All Tokens Are What You Need_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2404.07965)] [[GitHub](https://github.com/microsoft/rho)] [[Model (1B)](https://huggingface.co/microsoft/rho-math-1b-v0.1)] [[Model (7B)](https://huggingface.co/microsoft/rho-math-7b-v0.1)]

- **(MAmmoTH2)** _MAmmoTH2: Scaling Instructions from the Web_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2405.03548)] [[GitHub](https://github.com/TIGER-AI-Lab/MAmmoTH2)] [[Model (7B, Mistral)](https://huggingface.co/TIGER-Lab/MAmmoTH2-7B)] [[Model (8B, LLaMA-3)](https://huggingface.co/TIGER-Lab/MAmmoTH2-8B)] [[Model (8x7B, Mixtral)](https://huggingface.co/TIGER-Lab/MAmmoTH2-8x7B)]

- **(TheoremLlama)** _TheoremLlama: Transforming General-Purpose LLMs into Lean4 Experts_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2407.03203)] [[GitHub](https://github.com/RickySkywalker/TheoremLlama)] [[Model (8B)](https://huggingface.co/RickyDeSkywalker/TheoremLlama)]


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
[[Paper](https://arxiv.org/abs/2312.11370)] [[GitHub](https://github.com/pipilurj/G-LLaVA)] [[Model (7B)](https://huggingface.co/renjiepi/G-LLaVA-7B)] [[Model (13B)](https://huggingface.co/renjiepi/G-LLaVA-13B)]


<h3 id="mathematics-other-modalities-table">Other Modalities (Table)</h3>

- **(TAPAS)** _TAPAS: Weakly Supervised Table Parsing via Pre-training_ ```ACL 2020```     
[[Paper](https://arxiv.org/abs/2004.02349)] [[GitHub](https://github.com/google-research/tapas)] [[Model (Base)](https://storage.googleapis.com/tapas_models/2020_04_21/tapas_base.zip)] [[Model (Large)](https://storage.googleapis.com/tapas_models/2020_04_21/tapas_large.zip)]

- **(TaBERT)** _TaBERT: Learning Contextual Representations for Natural Language Utterances and Structured Tables_ ```ACL 2020```     
[[Paper](https://arxiv.org/abs/2005.08314)] [[GitHub](https://github.com/facebookresearch/TaBERT)] [[Model (Base)](https://drive.google.com/drive/folders/1fDW9rLssgDAv19OMcFGgFJ5iyd9p7flg)] [[Model (Large)](https://drive.google.com/drive/folders/1fDW9rLssgDAv19OMcFGgFJ5iyd9p7flg)]

- **(GraPPa)** _GraPPa: Grammar-Augmented Pre-training for Table Semantic Parsing_ ```ICLR 2021```     
[[Paper](https://arxiv.org/abs/2009.13845)] [[GitHub](https://github.com/taoyds/grappa)] [[Model (355M)](https://huggingface.co/Salesforce/grappa_large_jnt)]

- **(TUTA)** _TUTA: Tree-Based Transformers for Generally Structured Table Pre-training_ ```KDD 2021```     
[[Paper](https://arxiv.org/abs/2010.12537)] [[GitHub](https://github.com/microsoft/TUTA_table_understanding)]

- **(RCI)** _Capturing Row and Column Semantics in Transformer Based Question Answering over Tables_ ```NAACL 2021```     
[[Paper](https://arxiv.org/abs/2104.08303)] [[GitHub](https://github.com/IBM/row-column-intersection)] [[Model (12M)](https://huggingface.co/michaelrglass/albert-base-rci-wikisql-row)]

- **(TABBIE)** _TABBIE: Pretrained Representations of Tabular Data_ ```NAACL 2021```     
[[Paper](https://arxiv.org/abs/2105.02584)] [[GitHub](https://github.com/SFIG611/tabbie)]

- **(TAPEX)** _TAPEX: Table Pre-training via Learning a Neural SQL Executor_ ```ICLR 2022```     
[[Paper](https://arxiv.org/abs/2107.07653)] [[GitHub](https://github.com/microsoft/Table-Pretraining)] [[Model (140M)](https://huggingface.co/microsoft/tapex-base)] [[Model (406M)](https://huggingface.co/microsoft/tapex-large)]

- **(FORTAP)** _FORTAP: Using Formulas for Numerical-Reasoning-Aware Table Pretraining_ ```ACL 2022```     
[[Paper](https://arxiv.org/abs/2109.07323)] [[GitHub](https://github.com/microsoft/TUTA_table_understanding)]

- **(OmniTab)** _OmniTab: Pretraining with Natural and Synthetic Data for Few-shot Table-Based Question Answering_ ```NAACL 2022```     
[[Paper](https://arxiv.org/abs/2207.03637)] [[GitHub](https://github.com/jzbjyb/OmniTab)] [[Model (406M)](https://huggingface.co/neulab/omnitab-large)]

- **(ReasTAP)** _ReasTAP: Injecting Table Reasoning Skills During Pre-training via Synthetic Reasoning Examples_ ```EMNLP 2022```     
[[Paper](https://arxiv.org/abs/2210.12374)] [[GitHub](https://github.com/Yale-LILY/ReasTAP)] [[Model (406M)](https://huggingface.co/Yale-LILY/reastap-large)]

- **(Table-GPT)** _Table-GPT: Table-tuned GPT for Diverse Table Tasks_ ```SIGMOD 2024```     
[[Paper](https://arxiv.org/abs/2310.09263)]

- **(TableLlama)** _TableLlama: Towards Open Large Generalist Models for Tables_ ```NAACL 2024```     
[[Paper](https://arxiv.org/abs/2311.09206)] [[GitHub](https://github.com/OSU-NLP-Group/TableLlama)] [[Model (7B)](https://huggingface.co/osunlp/TableLlama)]

- **(TableLLM)** _TableLLM: Enabling Tabular Data Manipulation by LLMs in Real Office Usage Scenarios_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2403.19318)] [[GitHub](https://github.com/RUCKBReasoning/TableLLM)] [[Model (7B)](https://huggingface.co/RUCKBReasoning/TableLLM-7b)] [[Model (13B)](https://huggingface.co/RUCKBReasoning/TableLLM-13b)]





## Physics
<h3 id="physics-language">Language</h3>

- **(astroBERT)** _Building astroBERT, a Language Model for Astronomy & Astrophysics_ ```arXiv 2021```     
[[Paper](https://arxiv.org/abs/2112.00590)] [[Model (Base)](https://huggingface.co/adsabs/astroBERT)]

- **(AstroLLaMA)** _AstroLLaMA: Towards Specialized Foundation Models in Astronomy_ ```AACL 2023 Workshop```     
[[Paper](https://arxiv.org/abs/2309.06126)] [[Model (7B)](https://huggingface.co/universeTBD/astrollama)]

- **(AstroLLaMA-Chat)** _AstroLLaMA-Chat: Scaling AstroLLaMA with Conversational and Diverse Datasets_ ```Research Notes of the AAS 2024```     
[[Paper](https://arxiv.org/abs/2401.01916)] [[Model (7B)](https://huggingface.co/spaces/universeTBD/astrollama-7b-chat-alpha)]

- **(PhysBERT)** _PhysBERT: A Text Embedding Model for Physics Scientific Literature_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2408.09574)] [[Model (Base)](https://huggingface.co/thellert/physbert_cased)]





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

- **(Recycle-BERT)** _Recycle-BERT: Extracting Knowledge about Plastic Waste Recycling by Natural Language Processing_ ```ACS Sustainable Chemistry & Engineering 2023```     
[[Paper](https://pubs.acs.org/doi/10.1021/acssuschemeng.3c03162)] [[GitHub](https://github.com/avanscholar/Recycle_BERT_QandA)]

- **(CatBERTa)** _Catalyst Property Prediction with CatBERTa: Unveiling Feature Exploration Strategies through Large Language Models_ ```ACS Catalysis 2023```     
[[Paper](https://arxiv.org/abs/2309.00563)] [[GitHub](https://github.com/hoon-ock/CatBERTa)]

- **(LLM-Prop)** _LLM-Prop: Predicting Physical and Electronic Properties of Crystalline Solids from Their Text Descriptions_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2310.14029)] [[GitHub](https://github.com/vertaix/LLM-Prop)]

- **(ChemDFM)** _ChemDFM: Dialogue Foundation Model for Chemistry_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2401.14818)] [[GitHub](https://github.com/OpenDFM/ChemDFM)] [[Model (13B)](https://huggingface.co/OpenDFM/ChemDFM-13B-v1.0)]

- **(CrystalLLM)** _Fine-Tuned Language Models Generate Stable Inorganic Materials as Text_ ```ICLR 2024```     
[[Paper](https://arxiv.org/abs/2402.04379)] [[GitHub](https://github.com/facebookresearch/crystal-llm)]

- **(ChemLLM)** _ChemLLM: A Chemical Large Language Model_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2402.06852)] [[Model (7B)](https://huggingface.co/AI4Chem/ChemLLM-7B-Chat)]

- **(LlaSMol)** _LlaSMol: Advancing Large Language Models for Chemistry with a Large-Scale, Comprehensive, High-Quality Instruction Tuning Dataset_ ```COLM 2024```     
[[Paper](https://arxiv.org/abs/2402.09391)] [[GitHub](https://github.com/OSU-NLP-Group/LLM4Chem)] [[Model (6.7B, Galactica)](https://huggingface.co/osunlp/LlaSMol-Galactica-6.7B)] [[Model (7B, LLaMA-2)](https://huggingface.co/osunlp/LlaSMol-Llama2-7B)] [[Model (7B, Mistral)](https://huggingface.co/osunlp/LlaSMol-Mistral-7B)]


<h3 id="chemistry-language-graph">Language + Graph</h3>

- **(Text2Mol)** _Text2Mol: Cross-Modal Molecule Retrieval with Natural Language Queries_ ```EMNLP 2021```     
[[Paper](https://aclanthology.org/2021.emnlp-main.47)] [[GitHub](https://github.com/cnedwards/text2mol)]

- **(KV-PLM)** _A Deep-learning System Bridging Molecule Structure and Biomedical Text with Comprehension Comparable to Human Professionals_ ```Nature Communications 2022```     
[[Paper](https://www.nature.com/articles/s41467-022-28494-3)] [[GitHub](https://github.com/thunlp/KV-PLM)] [[Model (Base)](https://drive.google.com/drive/folders/1xig3-3JG63kR-Xqj1b9wkPEdxtfD_4IX)]

- **(MolT5)** _Translation between Molecules and Natural Language_ ```EMNLP 2022```     
[[Paper](https://arxiv.org/abs/2204.11817)] [[GitHub](https://github.com/blender-nlp/MolT5)] [[Model (60M)](https://huggingface.co/laituan245/molt5-small)] [[Model (220M)](https://huggingface.co/laituan245/molt5-base)] [[Model (770M)](https://huggingface.co/laituan245/molt5-large)]

- **(MoMu)** _A Molecular Multimodal Foundation Model Associating Molecule Graphs with Natural Language_ ```arXiv 2022```     
[[Paper](https://arxiv.org/abs/2209.05481)] [[GitHub](https://github.com/bingsu12/momu)]

- **(MoleculeSTM)** _Multi-modal Molecule Structure-text Model for Text-Based Retrieval and Editing_ ```Nature Machine Intelligence 2023```     
[[Paper](https://arxiv.org/abs/2212.10789)] [[GitHub](https://github.com/chao1224/MoleculeSTM)]

- **(Text+Chem T5)** _Unifying Molecular and Textual Representations via Multi-task Language Modelling_ ```ICML 2023```     
[[Paper](https://arxiv.org/abs/2301.12586)] [[GitHub](https://github.com/GT4SD/gt4sd-core)] [[Model (60M)](https://huggingface.co/GT4SD/multitask-text-and-chemistry-t5-small-augm)] [[Model (220M)](https://huggingface.co/GT4SD/multitask-text-and-chemistry-t5-base-augm)]

- **(GIMLET)** _GIMLET: A Unified Graph-Text Model for Instruction-Based Molecule Zero-Shot Learning_ ```NeurIPS 2023```     
[[Paper](https://arxiv.org/abs/2306.13089)] [[GitHub](https://github.com/zhao-ht/GIMLET)] [[Model (60M)](https://huggingface.co/haitengzhao/gimlet)]

- **(MolFM)** _MolFM: A Multimodal Molecular Foundation Model_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2307.09484)] [[GitHub](https://github.com/PharMolix/OpenBioMed)]

- **(MolCA)** _MolCA: Molecular Graph-Language Modeling with Cross-Modal Projector and Uni-Modal Adapter_ ```EMNLP 2023```     
[[Paper](https://arxiv.org/abs/2310.12798)] [[GitHub](https://github.com/acharkq/MolCA)]

- **(InstructMol)** _InstructMol: Multi-Modal Integration for Building a Versatile and Reliable Molecular Assistant in Drug Discovery_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2311.16208)] [[GitHub](https://github.com/IDEA-XL/InstructMol)]

- **(3D-MoLM)** _Towards 3D Molecule-Text Interpretation in Language Models_ ```ICLR 2024```     
[[Paper](https://arxiv.org/abs/2401.13923)] [[GitHub](https://github.com/lsh0520/3D-MoLM)]


<h3 id="chemistry-language-vision">Language + Vision</h3>

- **(GIT-Mol)** _GIT-Mol: A Multi-modal Large Language Model for Molecular Science with Graph, Image, and Text_ ```Computers in Biology and Medicine 2024```     
[[Paper](https://arxiv.org/abs/2308.06911)] [[GitHub](https://github.com/ai-hpc-research-team/git-mol)]


<h3 id="chemistry-other-modalities-molecule">Other Modalities (Molecule)</h3>

- **(SMILES-BERT)** _SMILES-BERT: Large Scale Unsupervised Pre-training for Molecular Property Prediction_ ```ACM BCB 2019```     
[[Paper](https://dl.acm.org/doi/abs/10.1145/3307339.3342186)] [[GitHub](https://github.com/uta-smile/SMILES-BERT)]

- **(MAT)** _Molecule Attention Transformer_ ```arXiv 2020```     
[[Paper](https://arxiv.org/abs/2002.08264)] [[GitHub](https://github.com/ardigen/MAT)]

- **(ChemBERTa)** _ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction_ ```arXiv 2020```     
[[Paper](https://arxiv.org/abs/2010.09885)] [[GitHub](https://github.com/seyonechithrananda/bert-loves-chemistry)] [[Model (125M)](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1)]

- **(MolBERT)** _Molecular Representation Learning with Language Models and Domain-Relevant Auxiliary Tasks_ ```arXiv 2020```     
[[Paper](https://arxiv.org/abs/2011.13230)] [[GitHub](https://github.com/BenevolentAI/MolBERT)] [[Model (Base)](https://ndownloader.figshare.com/files/25611290)]

- **(rxnfp)** _Mapping the Space of Chemical Reactions using Attention-Based Neural Networks_ ```Nature Machine Intelligence 2021```     
[[Paper](https://arxiv.org/abs/2012.06051)] [[GitHub](https://github.com/rxn4chemistry/rxnfp)] [[Model (Base)](https://github.com/rxn4chemistry/rxnfp/tree/master/rxnfp/models/transformers/bert_pretrained)]

- **(RXNMapper)** _Extraction of Organic Chemistry Grammar from Unsupervised Learning of Chemical Reactions_ ```Science Advances 2021```     
[[Paper](https://www.science.org/doi/10.1126/sciadv.abe4166)] [[GitHub](https://github.com/rxn4chemistry/rxnmapper)]

- **(MoLFormer)** _Large-Scale Chemical Language Representations Capture Molecular Structure and Properties_ ```Nature Machine Intelligence 2022```     
[[Paper](https://arxiv.org/abs/2106.09553)] [[GitHub](https://github.com/IBM/molformer)] [[Model (47M)](https://huggingface.co/katielink/MoLFormer-XL)]

- **(Chemformer)** _Chemformer: A Pre-trained Transformer for Computational Chemistry_ ```Machine Learning: Science and Technology 2022```     
[[Paper](https://chemrxiv.org/engage/chemrxiv/article-details/60ee8a3eb95bdd06d062074b)] [[GitHub](https://github.com/MolecularAI/Chemformer)] [[Model (45M)](https://az.app.box.com/s/7eci3nd9vy0xplqniitpk02rbg9q2zcq/folder/144881804954)] [[Model (230M)](https://az.app.box.com/s/7eci3nd9vy0xplqniitpk02rbg9q2zcq/folder/144881806154)]

- **(R-MAT)** _Relative Molecule Self-Attention Transformer_ ```Journal of Cheminformatics 2024```     
[[Paper](https://arxiv.org/abs/2110.05841)] [[GitHub](https://github.com/gmum/huggingmolecules)]

- **(MolGPT)** _MolGPT: Molecular Generation using a Transformer-Decoder Model_ ```Journal of Chemical Information and Modeling 2022```     
[[Paper](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00600)] [[GitHub](https://github.com/devalab/molgpt)]

- **(T5Chem)** _Unified Deep Learning Model for Multitask Reaction Predictions with Explanation_ ```Journal of Chemical Information and Modeling 2022```     
[[Paper](https://pubs.acs.org/doi/10.1021/acs.jcim.1c01467)] [[GitHub](https://github.com/HelloJocelynLu/t5chem)]

- **(ChemGPT)** _Neural Scaling of Deep Chemical Models_ ```Nature Machine Intelligence 2023```     
[[Paper](https://chemrxiv.org/engage/chemrxiv/article-details/627bddd544bdd532395fb4b5)] [[Model (4.7M)](https://huggingface.co/ncfrey/ChemGPT-4.7M)] [[Model (19M)](https://huggingface.co/ncfrey/ChemGPT-19M)] [[Model (1.2B)](https://huggingface.co/ncfrey/ChemGPT-1.2B)]

- **(Uni-Mol)** _Uni-Mol: A Universal 3D Molecular Representation Learning Framework_ ```ICLR 2023```     
[[Paper](https://chemrxiv.org/engage/chemrxiv/article-details/6402990d37e01856dc1d1581)] [[GitHub](https://github.com/deepmodeling/Uni-Mol)]

- **(TransPolymer)** _TransPolymer: A Transformer-Based Language Model for Polymer Property Predictions_ ```npj Computational Materials 2023```     
[[Paper](https://arxiv.org/abs/2209.01307)] [[GitHub](https://github.com/ChangwenXu98/TransPolymer)]

- **(polyBERT)** _polyBERT: A Chemical Language Model to Enable Fully Machine-Driven Ultrafast Polymer Informatics_ ```Nature Communications 2023```     
[[Paper](https://arxiv.org/abs/2209.14803)] [[GitHub](https://github.com/Ramprasad-Group/polyBERT)] [[Model (86M)](https://huggingface.co/kuelumbus/polyBERT)]

- **(MFBERT)** _Large-Scale Distributed Training of Transformers for Chemical Fingerprinting_ ```Journal of Chemical Information and Modeling 2022```     
[[Paper](https://pubs.acs.org/doi/10.1021/acs.jcim.2c00715)] [[GitHub](https://github.com/GouldGroup/MFBERT)]

- **(SPMM)** _Bidirectional Generation of Structure and Properties Through a Single Molecular Foundation Model_ ```Nature Communications 2024```     
[[Paper](https://arxiv.org/abs/2211.10590)] [[GitHub](https://github.com/jinhojsk515/SPMM)]

- **(BARTSmiles)** _BARTSmiles: Generative Masked Language Models for Molecular Representations_ ```Journal of Chemical Information and Modeling 2024```     
[[Paper](https://arxiv.org/abs/2211.16349)] [[GitHub](https://github.com/YerevaNN/BARTSmiles)] [[Model (406M)](https://huggingface.co/gayane/BARTSmiles)]

- **(MolGen)** _Domain-Agnostic Molecular Generation with Self-feedback_ ```ICLR 2024```     
[[Paper](https://arxiv.org/abs/2301.11259)] [[GitHub](https://github.com/zjunlp/MolGen)] [[Model (406M, BART)](https://huggingface.co/zjunlp/MolGen-large)] [[Model (7B, LLaMA)](https://huggingface.co/zjunlp/MolGen-7b)]

- **(SELFormer)** _SELFormer: Molecular Representation Learning via SELFIES Language Models_ ```Machine Learning: Science and Technology 2023```     
[[Paper](https://arxiv.org/abs/2304.04662)] [[GitHub](https://github.com/HUBioDataLab/SELFormer)] [[Model (58M)](https://drive.google.com/file/d/1zuVAKXCMc-HZHQo9y3Hu5zmQy51FGduI/view)] [[Model (87M)](https://drive.google.com/file/d/1zuVAKXCMc-HZHQo9y3Hu5zmQy51FGduI/view)]

- **(PolyNC)** _PolyNC: A Natural and Chemical Language Model for the Prediction of Unified Polymer Properties_ ```Chemical Science 2024```     
[[Paper](https://pubs.rsc.org/en/Content/ArticleLanding/2023/SC/D3SC05079C)] [[GitHub](https://github.com/HKQiu/Unified_ML4Polymers)] [[Model (220M)](https://huggingface.co/hkqiu/PolyNC)]





## Biology and Medicine
**Acknowledgment: We referred to Wang et al.'s survey paper [_Pre-trained Language Models in Biomedical Domain: A Systematic Survey_](https://arxiv.org/abs/2110.05006) and He et al.'s survey paper [_Foundation Model for Advancing Healthcare: Challenges, Opportunities, and Future Directions_](https://arxiv.org/abs/2404.03264) when writing some parts of this section.**

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
[[Paper](https://academic.oup.com/jamia/article-abstract/27/12/1935/5943218)] [[GitHub](https://github.com/uf-hobi-informatics-lab/ClinicalTransformerNER)] [[Model (Base, BERT)](https://transformer-models.s3.amazonaws.com/mimiciii_bert_10e_128b.zip)] [[Model (125M, RoBERTa)](https://transformer-models.s3.amazonaws.com/mimiciii_roberta_10e_128b.zip)] [[Model (12M, ALBERT)](https://transformer-models.s3.amazonaws.com/mimiciii_albert_10e_128b.zip)] [[Model (Base, ELECTRA)](https://transformer-models.s3.amazonaws.com/mimiciii_electra_5e_128b.zip)] [[Model (Base, XLNet)](https://transformer-models.s3.amazonaws.com/mimiciii_xlnet_5e_128b.zip)] [[Model (149M, Longformer)](https://transformer-models.s3.amazonaws.com/mimiciii_longformer_5e_128b.zip)] [[Model (86M, DeBERTa)](https://transformer-models.s3.amazonaws.com/mimiciii_deberta_10e_128b.tar.gz)]

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

- **(KeBioLM)** _Improving Biomedical Pretrained Language Models with Knowledge_ ```NAACL 2021 Workshop```     
[[Paper](https://arxiv.org/abs/2104.10344)] [[GitHub](https://github.com/GanjinZero/KeBioLM)]

- **(SciFive)** _SciFive: A Text-to-Text Transformer Model for Biomedical Literature_ ```arXiv 2021```     
[[Paper](https://arxiv.org/abs/2106.03598)] [[GitHub](https://github.com/justinphan3110/SciFive)] [[Model (220M)](https://huggingface.co/razent/SciFive-base-Pubmed_PMC)] [[Model (770M)](https://huggingface.co/razent/SciFive-large-Pubmed_PMC)]

- **(BioALBERT)** _Benchmarking for Biomedical Natural Language Processing Tasks with a Domain Specific ALBERT_ ```BMC Bioinformatics 2022```     
[[Paper](https://arxiv.org/abs/2107.04374)] [[GitHub](https://github.com/usmaann/BioALBERT)] [[Model (12M)](https://drive.google.com/file/d/1SIBd_-GETHhMiZ7BgMdDPEUDjOjtN_bH/view)] [[Model (18M)](https://drive.google.com/file/d/16KRtHf8Meze2Hcc4vK_GUNhG-9LY6_6P/view)]

- **(Clinical-Longformer)** _Clinical-Longformer and Clinical-BigBird: Transformers for Long Clinical Sequences_ ```arXiv 2022```     
[[Paper](https://arxiv.org/abs/2201.11838)] [[GitHub](https://github.com/luoyuanlab/Clinical-Longformer)] [[Model (149M, Longformer)](https://huggingface.co/yikuan8/Clinical-Longformer)] [[Model (Base, BigBird)](https://huggingface.co/yikuan8/Clinical-BigBird)]

- **(BioBART)** _BioBART: Pretraining and Evaluation of A Biomedical Generative Language Model_ ```ACL 2022 Workshop```     
[[Paper](https://arxiv.org/abs/2204.03905)] [[GitHub](https://github.com/GanjinZero/BioBART)] [[Model (140M)](https://huggingface.co/GanjinZero/biobart-base)] [[Model (406M)](https://huggingface.co/GanjinZero/biobart-large)]

- **(BioGPT)** _BioGPT: Generative Pre-trained Transformer for Biomedical Text Generation and Mining_ ```Briefings in Bioinformatics 2022```     
[[Paper](https://arxiv.org/abs/2210.10341)] [[GitHub](https://github.com/microsoft/BioGPT)] [[Model (355M)](https://huggingface.co/microsoft/biogpt)] [[Model (1.5B)](https://huggingface.co/microsoft/BioGPT-Large)]

- **(Med-PaLM)** _Large Language Models Encode Clinical Knowledge_ ```Nature 2023```     
[[Paper](https://arxiv.org/abs/2212.13138)]

- **(GatorTron)** _A Large Language Model for Electronic Health Records_ ```npj Digital Medicine 2022```     
[[Paper](https://www.nature.com/articles/s41746-022-00742-2)] [[GitHub](https://github.com/uf-hobi-informatics-lab/GatorTron)] [[Model (345M)](https://huggingface.co/UFNLP/gatortron-base)] [[Model (3.9B)](https://huggingface.co/UFNLP/gatortron-medium)] [[Model (8.9B)](https://huggingface.co/UFNLP/gatortron-large)]

- **(ChatDoctor)** _ChatDoctor: A Medical Chat Model Fine-Tuned on a Large Language Model Meta-AI (LLaMA) using Medical Domain Knowledge_ ```Cureus 2023```     
[[Paper](https://arxiv.org/abs/2303.14070)] [[GitHub](https://github.com/Kent0n-Li/ChatDoctor)]

- **(DoctorGLM)** _DoctorGLM: Fine-tuning your Chinese Doctor is not a Herculean Task_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2304.01097)] [[GitHub](https://github.com/xionghonglin/DoctorGLM)]

- **(BenTsao, f.k.a. HuaTuo)** _HuaTuo: Tuning LLaMA Model with Chinese Medical Knowledge_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2304.06975)] [[GitHub](https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese)]

- **(MedAlpaca)** _MedAlpaca - An Open-Source Collection of Medical Conversational AI Models and Training Data_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2304.08247)] [[GitHub](https://github.com/kbressem/medAlpaca)] [[Model (7B)](https://huggingface.co/medalpaca/medalpaca-7b)] [[Model (13B)](https://huggingface.co/medalpaca/medalpaca-13b)]

- **(PMC-LLaMA)** _PMC-LLaMA: Towards Building Open-source Language Models for Medicine_ ```JAMIA 2024```     
[[Paper](https://arxiv.org/abs/2304.14454)] [[GitHub](https://github.com/chaoyi-wu/PMC-LLaMA)] [[Model (7B)](https://huggingface.co/chaoyi-wu/PMC_LLAMA_7B)] [[Model (13B)](https://huggingface.co/axiong/PMC_LLaMA_13B)]

- **(Med-PaLM 2)** _Towards Expert-Level Medical Question Answering with Large Language Models_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2305.09617)]

- **(HuatuoGPT)** _HuatuoGPT, towards Taming Language Model to Be a Doctor_ ```EMNLP 2023 Findings```     
[[Paper](https://arxiv.org/abs/2305.15075)] [[GitHub](https://github.com/FreedomIntelligence/HuatuoGPT)] [[Model (7B)](https://huggingface.co/FreedomIntelligence/HuatuoGPT-7B)] [[Model (13B)](https://huggingface.co/FreedomIntelligence/HuatuoGPT-13b-delta)]

- **(MedCPT)** _MedCPT: Contrastive Pre-trained Transformers with Large-scale PubMed Search Logs for Zero-shot Biomedical Information Retrieval_ ```Bioinformatics 2023```     
[[Paper](https://arxiv.org/abs/2307.00589)] [[GitHub](https://github.com/ncbi/MedCPT)] [[Model (Base)](https://huggingface.co/ncbi/MedCPT-Query-Encoder)]

- **(Zhongjing)** _Zhongjing: Enhancing the Chinese Medical Capabilities of Large Language Model through Expert Feedback and Real-world Multi-turn Dialogue_ ```AAAI 2024```     
[[Paper](https://arxiv.org/abs/2308.03549)] [[GitHub](https://github.com/SupritYoung/Zhongjing)] [[Model (13B)](https://huggingface.co/Suprit/Zhongjing-LLaMA-base)]

- **(DISC-MedLLM)** _DISC-MedLLM: Bridging General Large Language Models and Real-World Medical Consultation_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2308.14346)] [[GitHub](https://github.com/FudanDISC/DISC-MedLLM)] [[Model (13B)](https://huggingface.co/Flmc/DISC-MedLLM)]

- **(DRG-LLaMA)** _DRG-LLaMA: Tuning LLaMA Model to Predict Diagnosis-related Group for Hospitalized Patients_ ```npj Digital Medicine 2024```     
[[Paper](https://arxiv.org/abs/2309.12625)] [[GitHub](https://github.com/hanyin88/DRG-LLaMA)]

- **(Qilin-Med)** _Qilin-Med: Multi-stage Knowledge Injection Advanced Medical Large Language Model_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2310.09089)] [[GitHub](https://github.com/williamliujl/Qilin-Med)]

- **(AlpaCare)** _AlpaCare: Instruction-tuned Large Language Models for Medical Application_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2310.14558)] [[GitHub](https://github.com/XZhang97666/AlpaCare)] [[Model (7B, LLaMA)](https://huggingface.co/xz97/AlpaCare-llama1-7b)] [[Model (7B, LLaMA-2)](https://huggingface.co/xz97/AlpaCare-llama2-7b)] [[Model (13B, LLaMA)](https://huggingface.co/xz97/AlpaCare-llama-13b)] [[Model (13B, LLaMA-2)](https://huggingface.co/xz97/AlpaCare-llama2-13b)]

- **(BianQue)** _BianQue: Balancing the Questioning and Suggestion Ability of Health LLMs with Multi-turn Health Conversations Polished by ChatGPT_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2310.15896)] [[GitHub](https://github.com/scutcyr/BianQue)] [[Model (6B)](https://huggingface.co/scutcyr/BianQue-1.0)]

- **(HuatuoGPT-II)** _HuatuoGPT-II, One-stage Training for Medical Adaption of LLMs_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2311.09774)] [[GitHub](https://github.com/FreedomIntelligence/HuatuoGPT-II)] [[Model (7B)](https://huggingface.co/FreedomIntelligence/HuatuoGPT2-7B)] [[Model (13B)](https://huggingface.co/FreedomIntelligence/HuatuoGPT2-13B)] [[Model (34B)](https://huggingface.co/FreedomIntelligence/HuatuoGPT2-34B)]

- **(Taiyi)** _Taiyi: A Bilingual Fine-Tuned Large Language Model for Diverse Biomedical Tasks_ ```JAMIA 2024```     
[[Paper](https://arxiv.org/abs/2311.11608)] [[GitHub](https://github.com/DUTIR-BioNLP/Taiyi-LLM)] [[Model (7B)](https://huggingface.co/DUTIR-BioNLP/Taiyi-LLM)]

- **(MEDITRON)** _MEDITRON-70B: Scaling Medical Pretraining for Large Language Models_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2311.16079)] [[GitHub](https://github.com/epfLLM/megatron-LLM)] [[Model (7B)](https://huggingface.co/epfl-llm/meditron-7b)] [[Model (70B)](https://huggingface.co/epfl-llm/meditron-70b)]

- **(PLLaMa)** _PLLaMa: An Open-source Large Language Model for Plant Science_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2401.01600)] [[GitHub](https://github.com/Xianjun-Yang/PLLaMa)] [[Model (7B)](https://huggingface.co/Xianjun/PLLaMa-7b-base)] [[Model (13B)](https://huggingface.co/Xianjun/PLLaMa-13b-base)]

- **(BioMistral)** _BioMistral: A Collection of Open-Source Pretrained Large Language Models for Medical Domains_ ```ACL 2024 Findings```     
[[Paper](https://arxiv.org/abs/2402.10373)] [[Model (7B)](https://huggingface.co/BioMistral/BioMistral-7B)]

- **(Me-LLaMA)** _Me-LLaMA: Foundation Large Language Models for Medical Applications_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2402.12749)] [[GitHub](https://github.com/BIDS-Xu-Lab/Me-LLaMA)]

- **(BiMediX)** _BiMediX: Bilingual Medical Mixture of Experts LLM_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2402.13253)] [[GitHub](https://github.com/mbzuai-oryx/BiMediX)] [[Model (8x7B)](https://huggingface.co/BiMediX/BiMediX-Bi)]

- **(MMedLM)** _Towards Building Multilingual Language Model for Medicine_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2402.13963)] [[GitHub](https://github.com/MAGIC-AI4Med/MMedLM)] [[Model (7B, InternLM)](https://huggingface.co/Henrychur/MMedLM)] [[Model (1.8B, InternLM2)](https://huggingface.co/Henrychur/MMedLM2-1_8B)] [[Model (7B, InternLM2)](https://huggingface.co/Henrychur/MMedLM2)] [[Model (8B, LLaMA-3)](https://huggingface.co/Henrychur/MMed-Llama-3-8B)]

- **(BioMedLM, f.k.a. PubMedGPT)** _BioMedLM: A 2.7B Parameter Language Model Trained On Biomedical Text_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2403.18421)] [[GitHub](https://github.com/stanford-crfm/BioMedLM)] [[Model (2.7B)](https://huggingface.co/stanford-crfm/BioMedLM)]

- **(Hippocrates)** _Hippocrates: An Open-Source Framework for Advancing Large Language Models in Healthcare_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2404.16621)] [[Model (7B, LLaMA-2)](https://huggingface.co/emrecanacikgoz/hippollama)] [[Model (7B, Mistral)](https://huggingface.co/emrecanacikgoz/hippomistral)]

- **(BMRetriever)** _BMRetriever: Tuning Large Language Models as Better Biomedical Text Retrievers_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2404.18443)] [[GitHub](https://github.com/ritaranx/BMRetriever)] [[Model (410M, Pythia)](https://huggingface.co/BMRetriever/BMRetriever-410M)] [[Model (1B, Pythia)](https://huggingface.co/BMRetriever/BMRetriever-1B)] [[Model (2B, Gemma)](https://huggingface.co/BMRetriever/BMRetriever-2B)] [[Model (7B, Mistral)](https://huggingface.co/BMRetriever/BMRetriever-7B)]

- **(Panacea)** _Panacea: A Foundation Model for Clinical Trial Search, Summarization, Design, and Recruitment_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2407.11007)] [[GitHub](https://github.com/linjc16/Panacea)]


<h3 id="biology-language-graph">Language + Graph</h3>

- **(G-BERT)** _Pre-training of Graph Augmented Transformers for Medication Recommendation_ ```IJCAI 2019```     
[[Paper](https://arxiv.org/abs/1906.00346)] [[GitHub](https://github.com/jshang123/G-Bert)]

- **(CODER)** _CODER: Knowledge Infused Cross-Lingual Medical Term Embedding for Term Normalization_ ```JBI 2022```     
[[Paper](https://arxiv.org/abs/2011.02947)] [[GitHub](https://github.com/GanjinZero/CODER)] [[Model (Base)](https://huggingface.co/GanjinZero/coder_eng)]

- **(MoP)** _Mixture-of-Partitions: Infusing Large Biomedical Knowledge Graphs into BERT_ ```EMNLP 2021```     
[[Paper](https://arxiv.org/abs/2109.04810)] [[GitHub](https://github.com/cambridgeltl/mop)]

- **(BioLinkBERT)** _LinkBERT: Pretraining Language Models with Document Links_ ```ACL 2022```     
[[Paper](https://arxiv.org/abs/2203.15827)] [[GitHub](https://github.com/michiyasunaga/LinkBERT)] [[Model (Base)](https://huggingface.co/michiyasunaga/BioLinkBERT-base)] [[Model (Large)](https://huggingface.co/michiyasunaga/BioLinkBERT-large)]

- **(DRAGON)** _Deep Bidirectional Language-Knowledge Graph Pretraining_ ```NeurIPS 2022```     
[[Paper](https://arxiv.org/abs/2210.09338)] [[GitHub](https://github.com/michiyasunaga/dragon)] [[Model (360M)](https://nlp.stanford.edu/projects/myasu/DRAGON/models/biomed_model.pt)]


<h3 id="biology-language-vision">Language + Vision</h3>

- **(ConVIRT)** _Contrastive Learning of Medical Visual Representations from Paired Images and Text_ ```MLHC 2022```     
[[Paper](https://arxiv.org/abs/2010.00747)] [[GitHub](https://github.com/yuhaozhang/convirt)]

- **(MMBERT)** _MMBERT: Multimodal BERT Pretraining for Improved Medical VQA_ ```ISBI 2021```     
[[Paper](https://arxiv.org/abs/2104.01394)] [[GitHub](https://github.com/VirajBagal/MMBERT)]

- **(MedViLL)** _Multi-modal Understanding and Generation for Medical Images and Text via Vision-Language Pre-training_ ```JBHI 2022```     
[[Paper](https://arxiv.org/abs/2105.11333)] [[GitHub](https://github.com/SuperSupermoon/MedViLL)]

- **(GLoRIA)** _GLoRIA: A Multimodal Global-Local Representation Learning Framework for Label-efficient Medical Image Recognition_ ```ICCV 2021```     
[[Paper](https://ieeexplore.ieee.org/document/9710099)] [[GitHub](https://github.com/marshuang80/gloria)]

- **(LoVT)** _Joint Learning of Localized Representations from Medical Images and Reports_ ```ECCV 2022```     
[[Paper](https://arxiv.org/abs/2112.02889)] [[GitHub](https://github.com/philip-mueller/lovt)]

- **(BioViL)** _Making the Most of Text Semantics to Improve Biomedical Vision-Language Processing_ ```ECCV 2022```     
[[Paper](https://arxiv.org/abs/2204.09817)] [[GitHub](https://github.com/microsoft/hi-ml/tree/main/hi-ml-multimodal)]

- **(M3AE)** _Multi-Modal Masked Autoencoders for Medical Vision-and-Language Pre-training_ ```MICCAI 2022```     
[[Paper](https://arxiv.org/abs/2209.07098)] [[GitHub](https://github.com/zhjohnchan/M3AE)] [[Model](https://drive.google.com/drive/folders/1b3_kiSHH8khOQaa7pPiX_ZQnUIBxeWWn)]

- **(ARL)** _Align, Reason and Learn: Enhancing Medical Vision-and-Language Pre-training with Knowledge_ ```ACM MM 2022```     
[[Paper](https://arxiv.org/abs/2209.07118)] [[GitHub](https://github.com/zhjohnchan/ARL)]

- **(CheXzero)** _Expert-Level Detection of Pathologies from Unannotated Chest X-ray Images via Self-Supervised Learning_ ```Nature Biomedical Engineering 2022```     
[[Paper](https://www.nature.com/articles/s41551-022-00936-9)] [[GitHub](https://github.com/rajpurkarlab/CheXzero)] [[Model](https://drive.google.com/drive/folders/1makFLiEMbSleYltaRxw81aBhEDMpVwno)]

- **(MGCA)** _Multi-Granularity Cross-modal Alignment for Generalized Medical Visual Representation Learning_ ```NeurIPS 2022```     
[[Paper](https://arxiv.org/abs/2210.06044)] [[GitHub](https://github.com/HKU-MedAI/MGCA)] [[Model](https://drive.google.com/drive/folders/15_mP9Lqq2H15R53qlKn3l_xzGVzi9jX9)]

- **(MedCLIP)** _MedCLIP: Contrastive Learning from Unpaired Medical Images and Text_ ```EMNLP 2022```     
[[Paper](https://arxiv.org/abs/2210.10163)] [[GitHub](https://github.com/RyanWangZf/MedCLIP)]

- **(BioViL-T)** _Learning to Exploit Temporal Structure for Biomedical Vision-Language Processing_ ```CVPR 2023```     
[[Paper](https://arxiv.org/abs/2301.04558)] [[GitHub](https://github.com/microsoft/hi-ml/tree/main/hi-ml-multimodal)] [[Model](https://huggingface.co/microsoft/BiomedVLP-BioViL-T)]

- **(BiomedCLIP)** _BiomedCLIP: A Multimodal Biomedical Foundation Model Pretrained from Fifteen Million Scientific Image-Text Pairs_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2303.00915)] [[Model](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)]

- **(PMC-CLIP)** _PMC-CLIP: Contrastive Language-Image Pre-training using Biomedical Documents_ ```MICCAI 2023```     
[[Paper](https://arxiv.org/abs/2303.07240)] [[GitHub](https://github.com/WeixiongLin/PMC-CLIP)] [[Model](https://huggingface.co/ryanyip7777/pmc_vit_l_14)]

- **(Xplainer)** _Xplainer: From X-Ray Observations to Explainable Zero-Shot Diagnosis_ ```MICCAI 2023```     
[[Paper](https://arxiv.org/abs/2303.13391)] [[GitHub](https://github.com/ChantalMP/Xplainer)]

- **(RGRG)** _Interactive and Explainable Region-Guided Radiology Report Generation_ ```CVPR 2023```     
[[Paper](https://arxiv.org/abs/2304.08295)] [[GitHub](https://github.com/ttanida/rgrg)] [[Model](https://drive.google.com/file/d/1rDxqzOhjqydsOrITJrX0Rj1PAdMeP7Wy/view)]

- **(BiomedGPT)** _A Generalist Vision-Language Foundation Model for Diverse Biomedical Tasks_ ```Nature Medicine 2024```     
[[Paper](https://arxiv.org/abs/2305.17100)] [[GitHub](https://github.com/taokz/BiomedGPT)] [[Model (33M)](https://www.dropbox.com/sh/cu2r5zkj2r0e6zu/AADZ-KHn-emsICawm9CM4MqVa?dl=0&e=1&preview=biomedgpt_tiny.pt)] [[Model (93M)](https://www.dropbox.com/sh/cu2r5zkj2r0e6zu/AADZ-KHn-emsICawm9CM4MqVa?dl=0&e=1&preview=biomedgpt_medium.pt)] [[Model (182M)](https://www.dropbox.com/sh/cu2r5zkj2r0e6zu/AADZ-KHn-emsICawm9CM4MqVa?dl=0&e=1&preview=biomedgpt_base.pt)]

- **(Med-UniC)** _Med-UniC: Unifying Cross-Lingual Medical Vision-Language Pre-Training by Diminishing Bias_ ```NeurIPS 2023```     
[[Paper](https://arxiv.org/abs/2305.19894)] [[GitHub](https://github.com/SUSTechBruce/Med-UniC)]

- **(LLaVA-Med)** _LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day_ ```NeurIPS 2023```     
[[Paper](https://arxiv.org/abs/2306.00890)] [[GitHub](https://github.com/microsoft/LLaVA-Med)] [[Model (7B)](https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b)]

- **(MI-Zero)** _Visual Language Pretrained Multiple Instance Zero-Shot Transfer for Histopathology Images_ ```CVPR 2023```     
[[Paper](https://arxiv.org/abs/2306.07831)] [[GitHub](https://github.com/mahmoodlab/MI-Zero)] [[Model](https://drive.google.com/drive/folders/1AR9agw2WLXes5wz26UTlT_mvJoUY38mQ)]

- **(XrayGPT)** _XrayGPT: Chest Radiographs Summarization using Medical Vision-Language Models_ ```ACL 2024 Workshop```     
[[Paper](https://arxiv.org/abs/2306.07971)] [[GitHub](https://github.com/mbzuai-oryx/XrayGPT)]

- **(MONET)** _Transparent Medical Image AI via an Image–Text Foundation Model Grounded in Medical Literature_ ```Nature Medicine 2024```     
[[Paper](https://www.medrxiv.org/content/10.1101/2023.06.07.23291119)] [[GitHub](https://github.com/suinleelab/MONET)]

- **(QuiltNet)** _Quilt-1M: One Million Image-Text Pairs for Histopathology_ ```NeurIPS 2023```     
[[Paper](https://arxiv.org/abs/2306.11207)] [[GitHub](https://github.com/wisdomikezogwo/quilt1m)] [[Model](https://huggingface.co/wisdomik/QuiltNet-B-16-PMB)]

- **(MUMC)** _Masked Vision and Language Pre-training with Unimodal and Multimodal Contrastive Losses for Medical Visual Question Answering_ ```MICCAI 2023```     
[[Paper](https://arxiv.org/abs/2307.05314)] [[GitHub](https://github.com/pengfeiliHEU/MUMC)]

- **(M-FLAG)** _M-FLAG: Medical Vision-Language Pre-training with Frozen Language Models and Latent Space Geometry Optimization_ ```MICCAI 2023```     
[[Paper](https://arxiv.org/abs/2307.08347)] [[GitHub](https://github.com/cheliu-computation/M-FLAG-MICCAI2023)]

- **(PRIOR)** _PRIOR: Prototype Representation Joint Learning from Medical Images and Reports_ ```ICCV 2023```     
[[Paper](https://arxiv.org/abs/2307.12577)] [[GitHub](https://github.com/QtacierP/PRIOR)]

- **(Med-PaLM M)** _Towards Generalist Biomedical AI_ ```NEJM AI 2024```     
[[Paper](https://arxiv.org/abs/2307.14334)] [[GitHub](https://github.com/kyegomez/Med-PaLM)]

- **(CITE)** _Text-Guided Foundation Model Adaptation for Pathological Image Classification_ ```MICCAI 2023```     
[[Paper](https://arxiv.org/abs/2307.14901)] [[GitHub](https://github.com/openmedlab/CITE)]

- **(Med-Flamingo)** _Med-Flamingo: A Multimodal Medical Few-shot Learner_ ```ML4H 2023```     
[[Paper](https://arxiv.org/abs/2307.15189)] [[GitHub](https://github.com/snap-stanford/med-flamingo)]

- **(RadFM)** _Towards Generalist Foundation Model for Radiology by Leveraging Web-scale 2D&3D Medical Data_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2308.02463)] [[GitHub](https://github.com/chaoyi-wu/RadFM)] [[Model](https://huggingface.co/chaoyi-wu/RadFM)]

- **(PLIP)** _A Visual–Language Foundation Model for Pathology Image Analysis using Medical Twitter_ ```Nature Medicine 2023```     
[[Paper](https://www.nature.com/articles/s41591-023-02504-3)] [[GitHub](https://github.com/PathologyFoundation/plip)] [[Model](https://huggingface.co/vinid/plip)]

- **(MaCo)** _Enhancing Representation in Radiography-Reports Foundation Model: A Granular Alignment Algorithm Using Masked Contrastive Learning_ ```Nature Communications 2024```     
[[Paper](https://arxiv.org/abs/2309.05904)] [[GitHub](https://github.com/SZUHvern/MaCo)]

- **(CXR-CLIP)** _CXR-CLIP: Toward Large Scale Chest X-ray Language-Image Pre-training_ ```MICCAI 2023```     
[[Paper](https://arxiv.org/abs/2310.13292)] [[GitHub](https://github.com/kakaobrain/cxr-clip)]

- **(Qilin-Med-VL)** _Qilin-Med-VL: Towards Chinese Large Vision-Language Model for General Healthcare_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2310.17956)] [[GitHub](https://github.com/williamliujl/Qilin-Med-VL)] [[Model](https://huggingface.co/williamliu/Qilin-Med-VL)]

- **(BioCLIP)** _BioCLIP: A Vision Foundation Model for the Tree of Life_ ```CVPR 2024```    
[[Paper](https://arxiv.org/abs/2311.18803)] [[GitHub](https://github.com/Imageomics/BioCLIP)] [[Model](https://huggingface.co/imageomics/bioclip)]

- **(M3D)** _M3D: Advancing 3D Medical Image Analysis with Multi-Modal Large Language Models_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2404.00578)] [[GitHub](https://github.com/BAAI-DCAI/M3D)] [[Model](https://huggingface.co/GoodBaiBai88/M3D-CLIP)]

- **(Med-Gemini)** _Capabilities of Gemini Models in Medicine_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2404.18416)]

- **(Med-Gemini-2D/3D/Polygenic)** _Advancing Multimodal Medical Capabilities of Gemini_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2405.03162)]

- **(Mammo-CLIP)** _Mammo-CLIP: A Vision Language Foundation Model to Enhance Data Efficiency and Robustness in Mammography_ ```MICCAI 2024```     
[[Paper](https://arxiv.org/abs/2405.12255)] [[GitHub](https://github.com/batmanlab/Mammo-CLIP)] [[Model](https://huggingface.co/shawn24/Mammo-CLIP)]


<h3 id="biology-other-modalities-protein">Other Modalities (Protein)</h3>

- **(ProtTrans)** _ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning_ ```TPAMI 2021```     
[[Paper](https://arxiv.org/abs/2007.06225)] [[GitHub](https://github.com/agemagician/ProtTrans)] [[Model (420M, BERT)](https://huggingface.co/Rostlab/prot_bert_bfd)] [[Model (224M, ALBERT)](https://huggingface.co/Rostlab/prot_albert)] [[Model (409M, XLNet)](https://huggingface.co/Rostlab/prot_xlnet)] [[Model (420M, ELECTRA)](https://huggingface.co/Rostlab/prot_electra_generator_bfd)] [[Model (3B, T5)](https://huggingface.co/Rostlab/prot_t5_xl_bfd)] [[Model (11B, T5)](https://huggingface.co/Rostlab/prot_t5_xxl_bfd)]

- **(ESM-1b)** _Biological Structure and Function Emerge from Scaling Unsupervised Learning to 250 Million Protein Sequences_ ```PNAS 2021```     
[[Paper](https://www.biorxiv.org/content/10.1101/622803)] [[GitHub](https://github.com/facebookresearch/esm)] [[Model (650M)](https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt)]

- **(MSA Transformer)** _MSA Transformer_ ```ICML 2021```     
[[Paper](https://www.biorxiv.org/content/10.1101/2021.02.12.430858)] [[GitHub](https://github.com/rmrao/msa-transformer)]

- **(ESM-1v)** _Language Models Enable Zero-Shot Prediction of the Effects of Mutations on Protein Function_ ```NeurIPS 2021```     
[[Paper](https://www.biorxiv.org/content/10.1101/2021.07.09.450648)] [[GitHub](https://github.com/facebookresearch/esm)] [[Model (650M)](https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_1.pt)]

- **(AminoBERT)** _Single-Sequence Protein Structure Prediction using a Language Model and Deep Learning_ ```Nature Biotechnology 2022```     
[[Paper](https://www.biorxiv.org/content/10.1101/2021.08.02.454840)] [[GitHub](https://github.com/aqlaboratory/rgn2)]

- **(ProteinBERT)** _ProteinBERT: A Universal Deep-Learning Model of Protein Sequence and Function_ ```Bioinformatics 2022```     
[[Paper](https://academic.oup.com/bioinformatics/article/38/8/2102/6502274)] [[GitHub](https://github.com/nadavbra/protein_bert)] [[Model (16M)](https://huggingface.co/GrimSqueaker/proteinBERT)]

- **(ProtGPT2)** _ProtGPT2 is a Deep Unsupervised Language Model for Protein Design_ ```Nature Communications 2022```     
[[Paper](https://www.biorxiv.org/content/10.1101/2022.03.09.483666)] [[Model (738M)](https://huggingface.co/nferruz/ProtGPT2)]

- **(ESM-IF1)** _Learning Inverse Folding from Millions of Predicted Structures_ ```ICML 2022```     
[[Paper](https://www.biorxiv.org/content/10.1101/2022.04.10.487779)] [[GitHub](https://github.com/facebookresearch/esm)] [[Model (142M)](https://dl.fbaipublicfiles.com/fair-esm/models/esm_if1_gvp4_t16_142M_UR50.pt)]

- **(ProGen)** _Large Language Models Generate Functional Protein Sequences across Diverse Families_ ```Nature Biotechnology 2023```     
[[Paper](https://www.nature.com/articles/s41587-022-01618-2)] [[GitHub](https://github.com/salesforce/progen)] [[Model (1.6B)](https://zenodo.org/records/7309036)]

- **(ProGen2)** _ProGen2: Exploring the Boundaries of Protein Language Models_ ```Cell Systems 2023```     
[[Paper](https://arxiv.org/abs/2206.13517)] [[GitHub](https://github.com/salesforce/progen)] [[Model (151M)](https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-small.tar.gz)] [[Model (764M)](https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-base.tar.gz)] [[Model (2.7B)](https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-large.tar.gz)] [[Model (6.4B)](https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-xlarge.tar.gz)]

- **(ESM-2)** _Evolutionary-Scale Prediction of Atomic-Level Protein Structure with a Language Model_ ```Science 2023```     
[[Paper](https://www.biorxiv.org/content/10.1101/2022.07.20.500902)] [[GitHub](https://github.com/facebookresearch/esm)] [[Model (8M)](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t6_8M_UR50D.pt)] [[Model (35M)](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t12_35M_UR50D.pt)] [[Model (150M)](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t30_150M_UR50D.pt)] [[Model (650M)](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt)] [[Model (3B)](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt)] [[Model (15B)](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t48_15B_UR50D.pt)]

- **(Ankh)** _Ankh: Optimized Protein Language Model Unlocks General-Purpose Modelling_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2301.06568)] [[GitHub](https://github.com/agemagician/Ankh)] [[Model (450M)](https://huggingface.co/ElnaggarLab/ankh-base)] [[Model (1.1B)](https://huggingface.co/ElnaggarLab/ankh-large)]

- **(ProtST)** _ProtST: Multi-Modality Learning of Protein Sequences and Biomedical Texts_ ```ICML 2023```     
[[Paper](https://arxiv.org/abs/2301.12040)] [[GitHub](https://github.com/DeepGraphLearning/ProtST)]

- **(LM-Design)** _Structure-informed Language Models Are Protein Designers_ ```ICML 2023```     
[[Paper](https://arxiv.org/abs/2302.01649)] [[GitHub](https://github.com/BytedProtein/ByProt)] [[Model (659M)](https://zenodo.org/records/10046338/files/lm_design_esm2_650m.zip)]

- **(ProteinDT)** _A Text-Guided Protein Design Framework_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2302.04611)] [[GitHub](https://github.com/chao1224/ProteinDT)]

- **(Prot2Text)** _Prot2Text: Multimodal Protein's Function Generation with GNNs and Transformers_ ```AAAI 2024```     
[[Paper](https://arxiv.org/abs/2307.14367)] [[GitHub](https://github.com/hadi-abdine/Prot2Text)] [[Model (256M)](https://1drv.ms/u/s!AhcBGHWGY2mukdh4KujJfOJ_PxvJog?e=C6x4E6)] [[Model (283M)](https://1drv.ms/u/s!AhcBGHWGY2mukdh1N1kfnmXBEar-Tw?e=fACWFt)] [[Model (398M)](https://1drv.ms/u/s!AhcBGHWGY2mukdh3yPy98rqWfYcTJA?e=ot1SX6)] [[Model (898M)](https://1drv.ms/u/s!AhcBGHWGY2mukdh2EL4iP_IoVKu1tg?e=PioL6B)]

- **(BioMedGPT)** _BioMedGPT: Open Multimodal Generative Pre-trained Transformer for BioMedicine_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2308.09442)] [[GitHub](https://github.com/PharMolix/OpenBioMed)] [[Model (10B)](https://huggingface.co/PharMolix/BioMedGPT-10B)]

- **(SaProt)** _SaProt: Protein Language Modeling with Structure-Aware Vocabulary_ ```ICLR 2024```     
[[Paper](https://www.biorxiv.org/content/10.1101/2023.10.01.560349)] [[GitHub](https://github.com/westlake-repl/SaProt)] [[Model (35M)](https://huggingface.co/westlake-repl/SaProt_35M_AF2)] [[Model (650M)](https://huggingface.co/westlake-repl/SaProt_650M_AF2)]

- **(BioT5)** _BioT5: Enriching Cross-modal Integration in Biology with Chemical Knowledge and Natural Language Associations_ ```EMNLP 2023```     
[[Paper](https://arxiv.org/abs/2310.07276)] [[GitHub](https://github.com/QizhiPei/BioT5)] [[Model (220M)](https://huggingface.co/QizhiPei/biot5-base)]

- **(ProLLaMA)** _ProLLaMA: A Protein Large Language Model for Multi-Task Protein Language Processing_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2402.16445)] [[GitHub](https://github.com/PKU-YuanGroup/ProLLaMA)] [[Model (7B)](https://huggingface.co/GreatCaptainNemo/ProLLaMA)]


<h3 id="biology-other-modalities-dna">Other Modalities (DNA)</h3>

- **(DNABERT)** _DNABERT: Pre-trained Bidirectional Encoder Representations from Transformers Model for DNA-Language in Genome_ ```Bioinformatics 2021```     
[[Paper](https://www.biorxiv.org/content/10.1101/2020.09.17.301879)] [[GitHub](https://github.com/jerryji1993/DNABERT)] [[Model (Base)](https://drive.google.com/file/d/1BJjqb5Dl2lNMg2warsFQ0-Xvn1xxfFXC/view)]

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

- **(DNAGPT)** _DNAGPT: A Generalized Pre-trained Tool for Versatile DNA Sequence Analysis Tasks_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2307.05628)] [[GitHub](https://github.com/TencentAILabHealthcare/DNAGPT)] [[Model (0.1B)](https://drive.google.com/file/d/15m6CH3zaMSqflOaf6ec5VPfiulg-Gh0u/view)] [[Model (3B)](https://drive.google.com/file/d/1pQ3Ai7C-ObzKkKTRwuf6eshVneKHzYEg/view)]


<h3 id="biology-other-modalities-rna">Other Modalities (RNA)</h3>

- **(RNABERT)** _Informative RNA-base Embedding for Functional RNA Structural Alignment and Clustering by Deep Representation Learning_ ```NAR Genomics and Bioinformatics 2022```     
[[Paper](https://www.biorxiv.org/content/10.1101/2021.08.23.457433)] [[GitHub](https://github.com/mana438/RNABERT)]

- **(RNA-FM)** _Interpretable RNA Foundation Model from Unannotated Data for Highly Accurate RNA Structure and Function Predictions_ ```arXiv 2022```     
[[Paper](https://arxiv.org/abs/2204.00300)] [[GitHub](https://github.com/ml4bio/RNA-FM)]

- **(SpliceBERT)** _Self-Supervised Learning on Millions of Primary RNA Sequences from 72 Vertebrates Improves Sequence-Based RNA Splicing Prediction_ ```Briefings in Bioinformatics 2024```     
[[Paper](https://www.biorxiv.org/content/10.1101/2023.01.31.526427)] [[GitHub](https://github.com/chenkenbio/SpliceBERT)] [[Model (19.4M)](https://zenodo.org/records/7995778)]

- **(RNA-MSM)** _Multiple Sequence-Alignment-Based RNA Language Model and its Application to Structural Inference_ ```Nucleic Acids Research 2024```     
[[Paper](https://www.biorxiv.org/content/10.1101/2023.03.15.532863)] [[GitHub](https://github.com/yikunpku/RNA-MSM)]

- **(CodonBERT)** _CodonBERT: Large Language Models for mRNA Design and Optimization_ ```bioRxiv 2023```     
[[Paper](https://www.biorxiv.org/content/10.1101/2023.09.09.556981)] [[GitHub](https://github.com/Sanofi-Public/CodonBERT)]

- **(UTR-LM)** _A 5' UTR Language Model for Decoding Untranslated Regions of mRNA and Function Predictions_ ```Nature Machine Intelligence 2024```     
[[Paper](https://www.biorxiv.org/content/10.1101/2023.10.11.561938)] [[GitHub](https://github.com/a96123155/UTR-LM)]


<h3 id="biology-other-modalities-multiomics">Other Modalities (Multiomics)</h3>

- **(scBERT)** _scBERT as a Large-scale Pretrained Deep Language Model for Cell Type Annotation of Single-cell RNA-seq Data_ ```Nature Machine Intelligence 2022```     
[[Paper](https://www.biorxiv.org/content/10.1101/2021.12.05.471261)] [[GitHub](https://github.com/TencentAILabHealthcare/scBERT)]

- **(scGPT)** _scGPT: Towards Building a Foundation Model for Single-Cell Multi-omics using Generative AI_ ```Nature Methods 2024```     
[[Paper](https://www.biorxiv.org/content/10.1101/2023.04.30.538439)] [[GitHub](https://github.com/bowang-lab/scGPT)]

- **(scFoundation)** _Large Scale Foundation Model on Single-cell Transcriptomics_ ```Nature Methods 2024```     
[[Paper](https://www.biorxiv.org/content/10.1101/2023.05.29.542705)] [[GitHub](https://github.com/biomap-research/scFoundation)] [[Model (100M)](https://hopebio2020-my.sharepoint.com/:f:/g/personal/dongsheng_biomap_com/Eh22AX78_AVDv6k6v4TZDikBXt33gaWXaz27U9b1SldgbA)]

- **(Geneformer)** _Transfer Learning Enables Predictions in Network Biology_ ```Nature 2023```     
[[Paper](https://www.nature.com/articles/s41586-023-06139-9)] [[Model (10M)](https://huggingface.co/ctheodoris/Geneformer/blob/main/pytorch_model.bin)] [[Model (40M)](https://huggingface.co/ctheodoris/Geneformer/blob/main/geneformer-12L-30M/pytorch_model.bin)]

- **(CellLM)** _Large-Scale Cell Representation Learning via Divide-and-Conquer Contrastive Learning_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2306.04371)] [[GitHub](https://github.com/PharMolix/OpenBioMed)]

- **(CellPLM)** _CellPLM: Pre-training of Cell Language Model Beyond Single Cells_ ```ICLR 2024```     
[[Paper](https://www.biorxiv.org/content/10.1101/2023.10.03.560734)] [[GitHub](https://github.com/OmicsML/CellPLM)] [[Model (82M)](https://www.dropbox.com/scl/fo/i5rmxgtqzg7iykt2e9uqm/h?rlkey=o8hi0xads9ol07o48jdityzv1&dl=0)]





## Geography, Geology, and Environmental Science
<h3 id="geography-language">Language</h3>

- **(ClimateBERT)** _ClimateBERT: A Pretrained Language Model for Climate-Related Text_ ```arXiv 2021```     
[[Paper](https://arxiv.org/abs/2110.12010)] [[GitHub](https://github.com/climatebert/language-model)] [[Model (82M)](https://huggingface.co/climatebert/distilroberta-base-climate-f)]

- **(SpaBERT)** _SpaBERT: A Pretrained Language Model from Geographic Data for Geo-Entity Representation_ ```EMNLP 2022 Findings```     
[[Paper](https://arxiv.org/abs/2210.12213)] [[GitHub](https://github.com/zekun-li/spabert)] [[Model (Base)](https://drive.google.com/file/d/1l44FY3DtDxzM_YVh3RR6PJwKnl80IYWB/view)] [[Model (Large)](https://drive.google.com/file/d/1LeZayTR92R5bu9gH_cGCwef7nnMX35cR/view)]

- **(MGeo)** _MGeo: Multi-Modal Geographic Pre-training Method_ ```SIGIR 2023```     
[[Paper](https://arxiv.org/abs/2301.04283)] [[GitHub](https://github.com/PhantomGrapes/MGeo)]

- **(K2)** _K2: A Foundation Language Model for Geoscience Knowledge Understanding and Utilization_ ```WSDM 2024```     
[[Paper](https://arxiv.org/abs/2306.05064)] [[GitHub](https://github.com/davendw49/k2)] [[Model (7B)](https://huggingface.co/daven3/k2-v1)]

- **(OceanGPT)** _OceanGPT: A Large Language Model for Ocean Science Tasks_ ```ACL 2024```     
[[Paper](https://arxiv.org/abs/2310.02031)] [[GitHub](https://github.com/zjunlp/KnowLM)] [[Model (7B)](https://huggingface.co/zjunlp/OceanGPT-7b)]

- **(ClimateBERT-NetZero)** _ClimateBERT-NetZero: Detecting and Assessing Net Zero and Reduction Targets_ ```EMNLP 2023```     
[[Paper](https://arxiv.org/abs/2310.08096)] [[Model (82M)](https://huggingface.co/climatebert/netzero-reduction)]

- **(GeoLM)** _GeoLM: Empowering Language Models for Geospatially Grounded Language Understanding_ ```EMNLP 2023```     
[[Paper](https://arxiv.org/abs/2310.14478)] [[GitHub](https://github.com/knowledge-computing/geolm)]

- **(GeoGalactica)** _GeoGalactica: A Scientific Large Language Model in Geoscience_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2401.00434)] [[GitHub](https://github.com/geobrain-ai/geogalactica)] [[Model (30B)](https://huggingface.co/geobrain-ai/geogalactica)]


<h3 id="geography-language-graph">Language + Graph</h3>

- **(ERNIE-GeoL)** _ERNIE-GeoL: A Geography-and-Language Pre-trained Model and its Applications in Baidu Maps_ ```KDD 2022```     
[[Paper](https://arxiv.org/abs/2203.09127)]

- **(PK-Chat)** _PK-Chat: Pointer Network Guided Knowledge Driven Generative Dialogue Model_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2304.00592)] [[GitHub](https://github.com/iiot-tbb/Dialogue_DDE)]


<h3 id="geography-language-vision">Language + Vision</h3>

- **(UrbanCLIP)** _UrbanCLIP: Learning Text-Enhanced Urban Region Profiling with Contrastive Language-Image Pretraining from the Web_ ```WWW 2024```     
[[Paper](https://arxiv.org/abs/2310.18340)] [[GitHub](https://github.com/stupidbuluchacha/urbanclip)]


<h3 id="geography-other-modalities-climate-time-series">Other Modalities (Climate Time Series)</h3>

- **(FourCastNet)** _FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators_ ```arXiv 2022```     
[[Paper](https://arxiv.org/abs/2202.11214)] [[GitHub](https://github.com/NVlabs/FourCastNet)]

- **(Pangu-Weather)** _Accurate Medium-Range Global Weather Forecasting with 3D Neural Networks_ ```Nature 2023```     
[[Paper](https://arxiv.org/abs/2211.02556)] [[GitHub](https://github.com/198808xc/Pangu-Weather)]

- **(ClimaX)** _ClimaX: A Foundation Model for Weather and Climate_ ```ICML 2023```     
[[Paper](https://arxiv.org/abs/2301.10343)] [[GitHub](https://github.com/microsoft/ClimaX)]

- **(FengWu)** _FengWu: Pushing the Skillful Global Medium-Range Weather Forecast beyond 10 Days Lead_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2304.02948)] [[GitHub](https://github.com/OpenEarthLab/FengWu)]

- **(W-MAE)** _W-MAE: Pre-trained Weather Model with Masked Autoencoder for Multi-Variable Weather Forecasting_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2304.08754)] [[GitHub](https://github.com/gufrannn/w-mae)]

- **(FuXi)** _FuXi: A Cascade Machine Learning Forecasting System for 15-day Global Weather Forecast_ ```npj Climate and Atmospheric Science 2023```     
[[Paper](https://arxiv.org/abs/2306.12873)] [[GitHub](https://github.com/tpys/FuXi)]





## Citation
If you find this repository useful, please cite the following paper:
```
@article{zhang2024comprehensive,
  title={A Comprehensive Survey of Scientific Large Language Models and Their Applications in Scientific Discovery},
  author={Zhang, Yu and Chen, Xiusi and Jin, Bowen and Wang, Sheng and Ji, Shuiwang and Wang, Wei and Han, Jiawei},
  journal={arXiv preprint arXiv:2406.10833},
  year={2024}
}
```
