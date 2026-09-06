# SKILLS_INDEX.md

Purpose: Route tasks to installed skills with minimal context.

Rule:
Before acting, check this index. If a task matches a skill group, read only the relevant skill docs or invoke the relevant app/tool. Do not load unrelated skills.

## Core authoring / build skills

| Task | Use |
|---|---|
| Compile LaTeX / TeX / .tex to PDF | Latex Tectonic |
| Fix LaTeX compile errors | Latex Tectonic |
| Bibliography, equations, TeX build pipeline | Latex Tectonic |

## Research apps

| Task | Use |
|---|---|
| Save/read/search highlights, notes, reading workflow | Readwise |
| Scientific answers with citation context | Scite |

## Life-sciences meta-router

Use **Research Router** when the request is broad, ambiguous, or spans multiple biomedical databases.

Examples:
- “Find evidence linking gene X to disease Y”
- “Summarize biology of target X”
- “Map this GWAS locus to candidate genes”
- “Find variants, expression, drugs, pathways for this gene”

If the task is specific, route directly below.

## Literature / publications

| Task | Use |
|---|---|
| Peer-reviewed scientific evidence / citation support | Scite |
| Preprints | bioRxiv / medRxiv |
| PMC open-access articles | NCBI PMC |
| General biomedical metadata/search | NCBI Entrez |

## Genes, proteins, variants

| Task | Use |
|---|---|
| Gene / protein summary | UniProt, Ensembl, NCBI Datasets |
| Protein expression / tissue localization | Human Protein Atlas |
| Protein structure prediction | AlphaFold |
| Experimental structure | RCSB PDB |
| Variant clinical significance | ClinVar / NCBI Variation |
| Population variant frequency / constraint | gnomAD |
| European variation data | EVA |
| Allele lookup | IPD |

## Disease, phenotype, GWAS, PheWAS

| Task | Use |
|---|---|
| GWAS studies / associations | GWAS Catalog |
| GWAS locus to candidate gene | Locus-to-Gene Mapper |
| Disease/target evidence | Open Targets |
| Cancer variant interpretation | CIViC |
| Cancer mutations / cohorts | cBioPortal |
| FinnGen PheWAS | FinnGen PheWAS |
| BioBank Japan PheWAS | BioBank Japan PheWAS |
| TPMI PheWAS | TPMI PheWAS |
| UKB-TOPMed PheWAS | UKB-TOPMed PheWAS |
| Gene burden associations | Genebass Gene Burden |
| Ontology term resolution | EFO Ontology |

## Expression, regulation, single-cell

| Task | Use |
|---|---|
| Tissue expression | Bgee, Human Protein Atlas |
| eQTL associations | GTEx eQTL, eQTL Catalogue |
| ENCODE regulatory datasets | ENCODE |
| Single-cell datasets | CELLxGENE |
| Functional annotations | QuickGO |

## Drugs, compounds, chemistry

| Task | Use |
|---|---|
| Compound summary | PubChem PUG, ChEBI |
| Drug/target activity | ChEMBL |
| Ligand-target binding | BindingDB |
| Pharmacogenomics | PharmGKB |
| Metabolites | HMDB |
| Biochemical reactions | Rhea |

## Pathways, networks, systems biology

| Task | Use |
|---|---|
| Pathways | Reactome |
| Protein interaction networks | STRING |
| Causal/epidemiological evidence graph | EpiGraphDB |

## Omics studies and datasets

| Task | Use |
|---|---|
| BioStudies / ArrayExpress | BioStudies / ArrayExpress |
| Proteomics studies | PRIDE, ProteomeXchange |
| Metabolomics studies | MetaboLights |
| Microbiome studies | MGnify |
| RNA families / noncoding RNA | RNAcentral |

## Sequence analysis

| Task | Use |
|---|---|
| Sequence similarity search | NCBI BLAST |
| Gene lookup tables | NCBI Clinical Tables |

## Routing principles

1. Prefer the most specific skill.
2. Use Research Router only when multiple biomedical sources may be needed.
3. Use Scite when the user asks for evidence quality, support/dispute context, or literature-backed answers.
4. Use NCBI Entrez for broad biomedical search when no narrower source is obvious.
5. Never load all skills. Load only the matched skill or router.