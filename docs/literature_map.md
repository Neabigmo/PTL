# Phase 01 Literature Map

Last updated: 2026-04-24 21:38 Asia/Shanghai

## Search Scope

This map supports the manuscript frame: perturbation prediction should be judged by whether predicted responses are transportable across biological contexts, not only by random-split expression fidelity.

Sources used:
- Scite literature MCP, searched on 2026-04-24.
- Local Zotero MCP was requested by the user, but no callable Zotero method was exposed in this session. This is logged as non-blocking.
- Preprints are included only when they are directly relevant to the fast-moving benchmark or foundation-model literature.

Search concepts:
- "virtual cell", "AI virtual cell", "single-cell foundation model", scGPT, Geneformer, CellFM, scFoundation.
- Perturb-seq, CROP-seq, sci-Plex, single-cell CRISPR screen, scPerturb.
- GEARS, CPA, chemCPA, scGen, CellOT, CINEMA-OT, perturbation prediction.
- calibration, selective prediction, abstention, conformal prediction, uncertainty quantification.
- transportability, external validity, domain generalization, out-of-distribution prediction.

## Evidence Map

| # | Category | Paper or resource | Year | Type | Core contribution | One-line relevance to PTL |
|---:|---|---|---:|---|---|---|
| 1 | Virtual cell | Bunne, Roohani, Rosen et al., "How to build the virtual cell with artificial intelligence" | 2024 | Peer-reviewed perspective | Defines an AI virtual cell vision as multi-scale, multi-modal, predictive cell modeling. | Provides the field-level motivation for reliability and transportability testing of virtual-cell predictions. |
| 2 | Virtual cell | Moraru, Schaff, Slepchenko et al., "Virtual Cell modelling and simulation software environment" | 2008 | Peer-reviewed method/resource | Describes VCell as a modular computational modeling and simulation environment. | Establishes that "virtual cell" historically meant mechanistic simulation before modern neural foundation models. |
| 3 | Virtual cell | Freddolino and Tavazoie, "The Dawn of Virtual Cell Biology" | 2012 | Peer-reviewed commentary | Discusses whole-cell modeling as a quantitative predictive systems-biology goal. | Frames virtual-cell modeling as prediction under incomplete mechanistic knowledge. |
| 4 | Single-cell foundation model | Cui, Wang, Maan et al., "scGPT: toward building a foundation model for single-cell multi-omics using generative AI" | 2024 | Peer-reviewed model | Introduces a generative transformer foundation model for single-cell tasks including perturbation prediction. | A central virtual-cell-style model whose claimed generality needs context-stress evaluation. |
| 5 | Single-cell foundation model | Theodoris, Xiao, Chopra et al., "Transfer learning enables predictions in network biology" | 2023 | Peer-reviewed model | Introduces Geneformer for transfer learning over single-cell transcriptomic representations. | Represents foundation-model transfer learning, but not a direct solution to perturbation transportability. |
| 6 | Single-cell foundation model | Zeng, Xie, Shangguan et al., "CellFM: a large-scale foundation model pre-trained on transcriptomics of 100 million human cells" | 2025 | Peer-reviewed model | Scales single-cell pretraining to 100 million human cells. | Reinforces that model scale is rising faster than reliability benchmarks for unseen contexts. |
| 7 | Single-cell foundation model | Qiu, Chen, Qin et al., "BioLLM: A Standardized Framework for Integrating and Benchmarking Single-Cell Foundation Models" | 2024 | Preprint benchmark/resource | Proposes a unified interface for applying and comparing scBERT, Geneformer, scFoundation, and scGPT. | Supports the need for standardized model-agnostic evaluation interfaces. |
| 8 | Single-cell foundation model | Kedzierska, Crawford, Amini et al., "Assessing the limits of zero-shot foundation models in single-cell biology" | 2023 | Preprint benchmark | Evaluates zero-shot limits of Geneformer and scGPT-style models. | Signals that pretrained representation quality alone does not guarantee reliable out-of-domain behavior. |
| 9 | Model benchmarking | Ahlmann-Eltze, Huber, Anders et al., "Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines" | 2025 | Peer-reviewed benchmark | Compares foundation and deep perturbation models to simple baselines. | Directly motivates rigorous baselines and calibrated claims rather than model-first storytelling. |
| 10 | Perturbation prediction | Lotfollahi, Wolf, Theis et al., "scGen predicts single-cell perturbation responses" | 2019 | Peer-reviewed model | Uses VAE latent arithmetic to predict perturbation responses across single-cell contexts. | Early exemplar of cross-context perturbation prediction that PTL can treat as a model-agnostic target. |
| 11 | Perturbation prediction | Lotfollahi, Klimovskaia Susmelj, De Donno et al., "Predicting cellular responses to complex perturbations in high-throughput screens" | 2023 | Peer-reviewed model | Introduces CPA for factorized perturbation and covariate response modeling. | Shows explicit handling of covariates, but still needs reliability scoring under unseen context stress. |
| 12 | Perturbation prediction | Roohani, Huang, Leskovec et al., "Predicting transcriptional outcomes of novel multigene perturbations with GEARS" | 2023 | Peer-reviewed model | Predicts transcriptional outcomes of unseen multigene perturbations using graph-enhanced structure. | A key baseline/model for unseen perturbation and combination stress tests. |
| 13 | Perturbation prediction | Bunne, Stark, Gut et al., "Learning single-cell perturbation responses using neural optimal transport" | 2023 | Peer-reviewed model | Uses neural optimal transport to map unpaired control and perturbed cell distributions. | Highlights distribution-level response prediction and the need to evaluate cell-state heterogeneity. |
| 14 | Perturbation effect estimation | Dong, Wang, Wei et al., "Causal identification of single-cell experimental perturbation effects with CINEMA-OT" | 2023 | Peer-reviewed method | Uses causal optimal transport for single-cell experimental perturbation effects. | Distinguishes effect estimation from extrapolation and clarifies PTL's role for unseen settings. |
| 15 | Chemical perturbation prediction | Hetzel, Boehm, Kilbertus et al., "Predicting Cellular Responses to Novel Drug Perturbations at a Single-Cell Resolution" | 2022 | Preprint model | Extends CPA ideas with chemical structure information for unseen drug responses. | Supports a perturbation novelty feature family in PTL. |
| 16 | Multimodal perturbation prediction | Inecik, Uhlmann, Lotfollahi et al., "MultiCPA: Multimodal Compositional Perturbation Autoencoder" | 2022 | Preprint model | Extends CPA to multimodal single-cell perturbation settings. | Shows that modality transfer adds another context axis for transportability. |
| 17 | Perturbation prediction | Wang, Liu, Zhao et al., "Modeling and predicting single-cell multi-gene perturbation responses with scLAMBDA" | 2024 | Preprint model | Targets multi-gene perturbation response prediction. | Useful as emerging evidence that combinatorial prediction is active but still reliability-limited. |
| 18 | Perturbation prediction | Xing and Yau, "GPerturb: Gaussian process modelling of single-cell perturbation data" | 2025 | Preprint model | Applies Gaussian process modeling to sparse single-cell perturbation data. | Adds uncertainty-aware modeling ideas relevant to PTL features, while still requiring validation. |
| 19 | Perturb-seq technology | Dixit, Parnas, Li et al., "Perturb-Seq: Dissecting Molecular Circuits with Scalable Single-Cell RNA Profiling of Pooled Genetic Screens" | 2016 | Peer-reviewed technology | Couples pooled CRISPR perturbations with single-cell RNA profiling. | Foundational experimental technology for perturbation-response benchmarks. |
| 20 | Perturb-seq technology | Adamson, Norman, Jost et al., "A Multiplexed Single-Cell CRISPR Screening Platform..." | 2016 | Peer-reviewed technology | Uses Perturb-seq for systematic dissection of the unfolded protein response. | Demonstrates transcriptome-rich pooled perturbation screens suitable for response modeling. |
| 21 | Perturb-seq technology | Datlinger, Rendeiro, Schmidl et al., "Pooled CRISPR screening with single-cell transcriptome readout" | 2017 | Peer-reviewed technology | Introduces CROP-seq style pooled CRISPR screening with single-cell transcriptome readout. | Expands perturbation assay designs that later benchmarks must harmonize. |
| 22 | Perturb-seq dataset | Norman, Horlbeck, Replogle et al., "Exploring genetic interaction manifolds constructed from rich single-cell phenotypes" | 2019 | Peer-reviewed dataset/study | Maps combinatorial genetic interactions using Perturb-seq. | A priority source for unseen-combination and genetic-interaction stress splits. |
| 23 | Chemical perturbation dataset | Srivatsan, McFaline-Figueroa, Ramani et al., "Massively multiplex chemical transcriptomics at single-cell resolution" | 2020 | Peer-reviewed technology/dataset | Introduces sci-Plex for multiplexed chemical transcriptomics. | Provides chemical perturbation diversity for drug/covariate transportability questions. |
| 24 | Perturb-seq dataset | Replogle, Saunders, Pogson et al., "Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq" | 2022 | Peer-reviewed dataset/study | Scales Perturb-seq toward genome-scale genotype-phenotype landscapes. | Priority Tier 1 dataset family for later benchmark construction. |
| 25 | Perturb-seq technology | Schraivogel, Gschwind, Milbank et al., "Targeted Perturb-seq enables genome-scale genetic screens in single cells" | 2020 | Peer-reviewed technology | Improves targeted readouts for genome-scale single-cell perturbation screens. | Informs dataset diversity and possible bias in readout design. |
| 26 | Perturbation resource | Peidli, Green, Shen et al., "scPerturb: harmonized single-cell perturbation data" | 2024 | Peer-reviewed resource | Harmonizes 44 public single-cell perturbation-response datasets. | Primary resource for Phase 02 and a natural substrate for context-stress benchmark splits. |
| 27 | Perturbation modeling review | Ji, Lotfollahi, Wolf et al., "Machine learning for perturbational single-cell omics" | 2021 | Peer-reviewed review | Reviews machine learning for perturbational single-cell omics and the need for standardized data/benchmarks. | Prefigures this project's benchmark-first framing. |
| 28 | Perturbation modeling review | Gavriilidis, Vasileiou, Orfanou et al., "A mini-review on perturbation modelling across single-cell omic modalities" | 2024 | Peer-reviewed review | Surveys perturbation modeling methods across single-cell omic modalities. | Provides current taxonomy of models and technologies for literature positioning. |
| 29 | Benchmark design | Chevalley, Roohani, Mehrjou et al., "A large-scale benchmark for network inference from single-cell perturbation data" | 2025 | Peer-reviewed benchmark | Benchmarks causal network inference from perturbational single-cell data. | Shows benchmark value, but focuses network inference rather than response transportability. |
| 30 | Benchmark design | Li, You, Liao et al., "A Systematic Comparison of Single-Cell Perturbation Response Prediction Models" | 2024 | Preprint benchmark | Compares single-cell perturbation response prediction models. | Useful contemporary benchmark, but PTL should add explicit transportability and abstention axes. |
| 31 | Benchmark design | Wenteler, Occhetta, Branson et al., "PertEval-scFM" | 2024 | Preprint benchmark | Benchmarks single-cell foundation models for perturbation effect prediction. | Supports the need for standardized perturbation evaluation of foundation models. |
| 32 | Benchmark design | Li, Gao, She et al., "Benchmarking AI Models for In Silico Gene Perturbation of Cells" | 2024 | Preprint benchmark | Evaluates in silico gene perturbation models. | Shows rapid benchmark activity, but not yet a model-agnostic transportability lens. |
| 33 | Benchmark design | Boylan, Solovyeva, Bouiller et al., "Single Cell Foundation Models Evaluation (scFME) for In-Silico Perturbation" | 2025 | Preprint benchmark | Proposes evaluation for foundation models in in-silico perturbation. | Reinforces need for context-specific testing and metric calibration in newer benchmarks. |
| 34 | Domain generalization | Zhou, Liu, Qiao et al., "Domain Generalization: A Survey" | 2021 | Survey/preprint | Reviews learning when target distributions are inaccessible during training. | Supplies the ML vocabulary for unseen context stress splits. |
| 35 | Domain generalization | Blanchard, Deshmukh, Dogan et al., "Domain Generalization by Marginal Transfer Learning" | 2017 | Preprint theory/method | Formalizes domain generalization as learning across source distributions for a new distribution. | Supports context-as-domain framing for biological transportability. |
| 36 | Domain generalization in single-cell | Zhong, Hou, Yao et al., "Domain generalization enables general cancer cell annotation..." | 2024 | Peer-reviewed method | Applies domain generalization to cancer cell annotation in scRNA/spatial transcriptomics. | Demonstrates that biological domain shift can be handled explicitly, even outside perturbation prediction. |
| 37 | OOD detection in single-cell | Theunissen, Mortier, Saeys et al., "Evaluation of out-of-distribution detection methods for data shifts in single-cell transcriptomics" | 2025 | Peer-reviewed benchmark | Evaluates OOD detection for shifts in single-cell annotation. | Provides a direct template for detecting unreliable single-cell predictions under data shifts. |
| 38 | External validity | Averitt, Ryan, Weng et al., "A conceptual framework for external validity" | 2021 | Peer-reviewed conceptual framework | Formalizes external validity concepts for biomedical informatics. | Helps define transportability as more than performance on held-out samples. |
| 39 | Calibration | Guo, Pleiss, Sun et al., "On Calibration of Modern Neural Networks" | 2017 | Preprint/ML classic | Shows modern neural networks can be miscalibrated and popularizes temperature scaling diagnostics. | Motivates post-hoc calibration of perturbation prediction confidence. |
| 40 | Selective prediction | Geifman and El-Yaniv, "Selective Classification for Deep Neural Networks" | 2017 | Preprint/ML classic | Develops deep selective classification with abstention/coverage tradeoffs. | Supplies the keep-or-abstain framing for PTL selective prediction. |
| 41 | OOD detection | Hendrycks and Gimpel, "A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks" | 2016 | Preprint/ML classic | Introduces maximum softmax probability as a simple baseline for error/OOD detection. | Provides a naive confidence baseline for PTL comparisons. |
| 42 | Conformal prediction | Angelopoulos and Bates, "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification" | 2021 | Preprint/tutorial | Explains conformal prediction as distribution-free uncertainty quantification. | Supports calibrated prediction-set and risk-control language for PTL. |
| 43 | Calibrated regression | Kuleshov, Fenner, Ermon, "Accurate Uncertainties for Deep Learning Using Calibrated Regression" | 2018 | Preprint/ML method | Calibrates regression uncertainty estimates. | Relevant because perturbation fidelity targets are continuous, not only classification labels. |
| 44 | Biological UQ | Portela, Banga, Matabuena et al., "Conformal prediction for uncertainty quantification in dynamic biological systems" | 2025 | Peer-reviewed method | Applies conformal prediction to dynamic biological systems. | Shows distribution-free calibration entering computational biology, supporting PTL's reliability layer. |

## Thematic Synthesis

### Virtual-cell framing

Older virtual-cell work was grounded in mechanistic simulation, modular mathematical models, and numerical solvers. Recent AI virtual-cell writing shifts the emphasis toward learned, multi-modal, multi-scale predictors that can represent and simulate cells across states. That shift creates a sharper evaluation problem: if a model predicts a response, the field needs a way to know whether that response is likely to transfer across cell type, state, perturbation, combination, dataset, or assay context.

### Perturbation data substrate

Perturb-seq, CROP-seq, sci-Plex, targeted Perturb-seq, and genome-scale Perturb-seq make high-dimensional perturbation-response benchmarking possible. scPerturb reduces format fragmentation by harmonizing many public perturbation datasets. These resources create enough breadth for context-stress splits, but the literature still tends to treat datasets as sources for training and aggregate evaluation rather than as structured domains for transportability testing.

### Model landscape

scGen, CPA, GEARS, CellOT, CINEMA-OT, chemCPA, scGPT, Geneformer, CellFM, and newer models cover latent vector arithmetic, covariate-factorized generative models, graph priors, optimal transport, causal treatment-effect estimation, chemical descriptors, and foundation-model pretraining. This variety supports a model-agnostic layer: PTL should not be another predictor, but a reliability lens that can sit above many predictors.

### Benchmark and reliability gap

Recent benchmarks increasingly compare perturbation prediction models and sometimes include simple baselines. However, the major missing axis is not another average metric; it is a systematic stress test of whether a response learned in one biological context remains valid in a different context. Calibration, OOD detection, conformal prediction, and selective classification provide the methodological vocabulary for a keep-or-abstain reliability layer, but they are not yet integrated into a perturbation transportability benchmark.

## Candidate Citation Backbone

- Virtual-cell motivation: Bunne et al. (2024), Moraru et al. (2008), Freddolino and Tavazoie (2012).
- Foundation models: Cui et al. (2024), Theodoris et al. (2023), Zeng et al. (2025), Ahlmann-Eltze et al. (2025).
- Perturbation models: Lotfollahi et al. (2019), Lotfollahi et al. (2023), Roohani et al. (2023), Bunne et al. (2023), Dong et al. (2023).
- Data and benchmark substrate: Dixit et al. (2016), Adamson et al. (2016), Norman et al. (2019), Srivatsan et al. (2020), Replogle et al. (2022), Peidli et al. (2024).
- Transportability/reliability framing: Zhou et al. (2021), Zhong et al. (2024), Theunissen et al. (2025), Guo et al. (2017), Geifman and El-Yaniv (2017), Angelopoulos and Bates (2021).

