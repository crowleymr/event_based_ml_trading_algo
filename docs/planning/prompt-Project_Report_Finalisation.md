# Project Report Finalisation
This project has two overarching objectives:
1. To complete a university assignment as described in "C:\Users\crowl\OneDrive\Education\C04443 Master of Artificial Intelligence\2026s8 - 32513 Advanced Data Analytics Algorithms\Assignment\a2_specification.pdf"
2. To create the basis of a income producing trading bot, that ingests publicly available data and makes intelligent trading decisions across a universe of stocks to generate an positive risk-adjusted return above the market

## Objective
The next immediate objective of this thread is to plan the execution of the remaining tasks to complete the university assignment. "C:\Users\crowl\OneDrive\Education\C04443 Master of Artificial Intelligence\2026s8 - 32513 Advanced Data Analytics Algorithms\Assignment\a2_specification.pdf" states that the following is required:
- Project report
- Jupyter python notebook
- Implementation Log

Your objective is not to complete any specific section yourself. But to outline the specific objectives necessary to complete each section in a markdown plan, including delegation to other agents both within ChatGPT and Codex and external providers like perplexity.ai. The plan must include:
* Requirements to achieve a "Full Mark (100%)" Rubric plus any additional requirements or sub-requirements for each deliverable
* A technical plan to deliver on each requirement and sub-requirement
* An execution plan that will detail what part will be carried out by what service and what level model power would be required to get an optimal result
* What additional presentational coding work is needed to generate the graphs, tables and charts needed to meet the report needs

## Thread Guidance
* You need to also consider any outstanding or clarifying questions that are not obvious or available to be found out by reviewing the codebase, past agent threads, or the assignment specification and rubric. Present those questions to me and I'll respond to them as part of my planning review phase.
* I will review and comment in the markdown plan using the markdown comment <!-- ... --> you will be expected to detect and respond to these comments. My answers to outstanding questions will also be highlighted in comments.
* If you need to delegate to an external agent please generate a well formatted and relevant prompt for me to copy+paste. This is especially true for perplexity.ai calls.
* Final results are almost available and due in the next 15mins
* Re-runs are currently out-of-scope as the project is due to be complete tonight if possible

## Project report requirements
We are taking a hybrid route, between "Option 1: Study a fundamental machine learning model" and "Option 2: Build a machine learning system to address a practical challenge". This means that I intend to solve a real-world problem; "The capacity of ML and RL systems to make profitable short-to-medium term investment decisions based on trading events and publicly available information"; but done with an academic rigor above what would be expected for an undergraduate and masters level student.
This means that the "Project report" needs to cover:
- A `Problem definition` that describes the `Business problem` and derives from it a `Research question` that this study investigates
- A deep academic `Literature` review showing common modelling approaches towards the `Business problem` and `Research question` in the literature.
- A `Method` section that shows and justifies:
    - End-to-end high level system architecture and tech stack diagram
    - Pragmatic choice of data sources (because I don't have access to CRISP), universe of stocks selected, related back to literature and citing possible methodology improvement in the future 
    - Data cleaning and preparation method, and ETL pipeline diagram, showing how data is downloaded/extracted, transformed, and loaded. Key features derived from rules or models, the regressor target, and how train-test splits are applied. Also related back to literature if possible.
    - The list of models trialed, briefly summarising how they operate, and why they are academically interesting from a literature perspective
    - How models are evaluated against each other
    - HPO method, and how the "best possible candidate" of each model category is presented in the final evaluation 
    - Final evaluation methodology, including backtesting, and risk profiling
- `Results`: This section needs to summarise what was observed from each phase including:
    - Raw data, observations before and after cleaning
    - Statistical description of each feature, and which model they were used for
    - HPO performance per model trained, and a brief commentary on likely reasons why each parameter was selected for the final model version
    - Matrix of Loss functions for each model in training vs validation, indicating the effectiveness of training
    - Overall comparative model performance including:
        - Performance table, with winners across each metric category
        - Diagram of the long-run total equity performance of each model in the backtest bench
        - Chart of the relative return vs volatility frontiers of each model, with the left-most model in bold
- `Discussion`: Analyse the results implications including
    - Academic implications
    - Personal financial implications
    - Known limitations and gaps
    - Future improvements and refinements to methodology
    - Future planned deployment efforts  

**Additional Guidance:**
- The paper is expected to be prepared as a docx, with APA referencing making use of the built-in citations process or citations are collected and input into a zotero repository and then added to the built-in citations process. Whichever approach is simpler to implement autonomously.
- Past research was prepared by perplexity.ai in "D:\repos\event_based_ml_trading_algo\docs\planning\archived\ML_TRADING_LITERATURE_REVIEW.md" however, a new literature review query is potentially warranted

## Notebook requirements
Regarding the jupyter/python notebook, it has a requirement to be "self-contained", we have sevaral options and I need advice on which is more defensible:
- Present the notebook as a pipeline that calls modules throughout the codebase, this ensures that it is aligned with our approach, but it is not self-contained
- Present a "toy" example, where simplified code and methods are presented, without a lot of the defensive guardrails added to the main codebase, and executed on a more limited sample
- Recreate the "true" pipeline that is self-contained, but is unlikely to be able to execute in time for shipping

Regardless of the high level approach, I want the jupyter/python notebook to be a reflection of the paper, containing sections that reflect charts and tables that relate to each part of the methodology and results including:
- Environment set-up
- Data download
- ETL preprocessing pipeline
- Models Tested against HPO grains and Training Harnesses
- Evaluation Classes and functions
- HPO Scheduling and execution
- Final Evaluation Results and Charts
- Links throughout the notebook to the true module locations in the repo (relative paths) or GitHub (if notebook is shared via Google Colab)
- Markdown descriptions should have an educational tone, explaining relevant key concepts of each component. Like an interactive textbook.

## Implementation Log requirements
I need a detailed history docx organised in chronological order of each decision, action, turn, and interaction between the active threads in this project and several external threads as well:
- The original online ChatGPT thread at https://chatgpt.com/c/6a9cb0b1-ab90-83ec-ab0b-56b9f2ffdd81
- Past threads from perplexity.ai https://www.perplexity.ai/search/2433a3b0-5d37-4244-b470-665cb382c6f6

This history should also be prefaced with a draft of the Implementation Log, and discussions should reference log items. I will probably re-write from scratch myself, but am likely to re-use some parts.