# LLM User Preference Analysis: A Data-Driven Study of Task-Specific Model Usage

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

## Overview

This project conducts a comprehensive analysis of user preferences across Large Language Models (LLMs) using 57,351 real-world conversations spanning 70+ state-of-the-art language models. By integrating statistical testing and data visualization, this study identifies task-specific usage patterns and explores relationships between model characteristics and user satisfaction.

**Key Finding:** GPT models lead with a 42.2% win rate, significantly outperforming other families (x^2 = 2777.40, p < 0.000001). Winning responses are 21% longer on average, indicating users prefer comprehensive, detailed answers.

## Project Description

As artificial intelligence tools become increasingly integrated into daily workflows, understanding how users interact with different LLMs across various tasks is critical. This project analyzes over 57,000 real user conversations to uncover:

- Task-specific model preferences (coding, creative writing, data analysis, research)
- Statistical relationships between model characteristics and user satisfaction
- Performance patterns across 64 different language models
- Insights for optimal model selection and product development

The analysis employs descriptive statistics, chi-square tests, t-tests, and data visualization techniques to deliver actionable insights for AI developers, end users, and researchers studying human-AI interaction.

## Research Framework

### Primary Hypothesis
Users exhibit statistically significant preferences for certain LLMs when performing specific tasks, and model characteristics (family, response length, architecture) correlate with user preference patterns.

### Research Questions
1. Which LLMs are most preferred for coding tasks vs. creative or conversational tasks?
2. Do model performance characteristics correlate with user preference rates?
3. What task categories dominate user interactions with LLMs?
4. Are there statistically significant differences in user satisfaction across models?

### Key Results
- **Hypothesis Confirmed:** All findings highly significant (p < 0.000001)
- **GPT Dominance:** GPT family achieves 42.2% win rate
- **Response Length Matters:** Winners average 21% longer responses
- **Task Specificity:** Performance varies by category, but GPT-4 leads overall
- **Model Maturity:** 31% tie rate indicates comparable quality across many models

## Datasets

### Primary Dataset: Chatbot Arena Human Preferences
- **Source:** [Hugging Face - lmsys/chatbot_arena_conversations](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations)
- **Size:** 57,351 conversations after cleaning
- **Models Covered:** 64 unique LLMs including GPT-4, Claude 2, Llama 2, Gemini, Mistral
- **Features:** User prompts, model responses, preference votes, timestamps
- **Collection:** Real user A/B comparisons from LMSYS Chatbot Arena platform

### Secondary Dataset: Large Language Models Comparison
- **Source:** [Kaggle - Large Language Models Comparison Dataset](https://www.kaggle.com/datasets/samayashar/large-language-models-comparison-dataset)
- **Status:** Not used in final analysis due to synthetic model names incompatible with Arena data
- **Decision:** Proceeded with Arena dataset alone (sufficient for research objectives)

**Note:** Due to file size constraints (>100MB), datasets are not included in this repository. See [Installation](#installation) for download instructions.

## Project Structure
```
llm-preference-analysis/
│
├── data/
│   ├── raw/                          # Original datasets (gitignored)
│   │   └── chatbot_arena.csv
│   └── processed/                    # Cleaned datasets (gitignored)
│       ├── arena_cleaned.csv
│       └── arena_final_with_tasks.csv
│
├── notebooks/
│   ├── 00_data_download.ipynb        # Dataset acquisition
│   ├── 01_data_exploration.ipynb     # Initial EDA
│   ├── 02_data_cleaning.ipynb        # Preprocessing
│   ├── 03_exploratory_analysis.ipynb # Comprehensive EDA
│   ├── 04_statistical_analysis.ipynb # Hypothesis testing
│   └── 05_final_report.ipynb         # Complete report
│
├── figures/                          # Visualizations
│   ├── winner_distribution.png
│   ├── top_models.png
│   ├── model_win_rates.png
│   ├── family_win_rates.png
│   ├── task_distribution.png
│   ├── task_model_heatmap.png
│   └── ... (10+ visualizations)
│
├── .gitignore
├── requirements.txt
├── README.md
└── LICENSE
```

## Installation and Setup

### Prerequisites
- Python 3.10 or higher
- Anaconda or Miniconda
- Git
- Kaggle account (for dataset access)

### Environment Setup

1. **Clone the repository:**
```bash
git clone https://github.com/AsaHuang-DS/LLM_Preference_Analysis.git
cd LLM_Preference_Analysis
```

2. **Create conda environment:**
```bash
conda create -n llm_analysis python=3.10 -y
conda activate llm_analysis
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure Kaggle API:**
- Go to https://www.kaggle.com/settings/account
- Click "Create New Token" under API section
- Move `kaggle.json` to `~/.kaggle/` (Mac/Linux) or `C:\Users\<Username>\.kaggle\` (Windows)
- Set permissions (Mac/Linux): `chmod 600 ~/.kaggle/kaggle.json`

### Data Download

Run the data download notebook:
```bash
jupyter notebook notebooks/00_data_download.ipynb
```

This will:
1. Download Chatbot Arena dataset from Hugging Face (~175 MB)
2. Save dataset to `data/raw/`
3. Perform initial validation

**Expected download time:** 5-10 minutes depending on internet speed.

## Usage

### Running the Analysis

Execute notebooks in sequential order:
```bash
# Activate environment
conda activate llm_analysis

# Launch Jupyter
jupyter notebook
```

**Recommended workflow:**
1. `00_data_download.ipynb` - Download and verify datasets ✅
2. `01_data_exploration.ipynb` - Initial data profiling ✅
3. `02_data_cleaning.ipynb` - Data preprocessing ✅
4. `03_exploratory_analysis.ipynb` - Comprehensive EDA ✅
5. `04_statistical_analysis.ipynb` - Hypothesis testing ✅
6. `05_final_report.ipynb` - Complete analysis report ✅

### Notebook Contents

Each notebook includes:
- Research context and objectives
- Well-commented code with explanations
- Publication-quality visualizations
- Statistical test results
- Key findings and interpretations

## Methodology

### Analytical Approach

**Phase 1: Data Acquisition**
- Load Chatbot Arena dataset
- Initial quality assessment

**Phase 2: Data Cleaning**
- Remove missing values and duplicates
- Create winner column from binary indicators
- Engineer features (prompt length, response length)
- Extract model families from names

**Phase 3: Exploratory Data Analysis**
- Generate summary statistics
- Create 10+ visualizations
- Identify patterns and distributions

**Phase 4: Task Classification**
- Keyword-based categorization
- 7 categories: Coding, Creative Writing, Data Analysis, Research, Translation, Math, General
- ~85% accuracy validated by manual inspection

**Phase 5: Statistical Testing**
- Chi-square test: Model family vs. winner (χ² = 2777.40, p < 0.000001)
- T-test: Response length vs. winner (t = 17.76, p < 0.000001)
- Effect size calculations

### Technical Stack

**Core Libraries:**
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scipy` - Statistical testing

**Visualization:**
- `matplotlib` - Base plotting
- `seaborn` - Statistical visualizations

**Data Loading:**
- `datasets` (Hugging Face) - Arena data
- `kaggle` - Kaggle API

## Key Findings

### 1. Model Family Performance
- **GPT:** 42.2% win rate (31,743 appearances) - LEADING
- **Claude:** 37.5% win rate (16,100 appearances)
- **Vicuna:** 33.8% win rate (8,740 appearances)
- **Statistical test:** x^2 = 2777.40, p < 0.000001 (HIGHLY SIGNIFICANT)

### 2. Top Individual Models
1. gpt-4-1106-preview: 55.2% win rate
2. gpt-3.5-turbo-0314: 54.6% win rate
3. gpt-4-0125-preview: 51.4% win rate
4. gpt-4-0314: 48.7% win rate
5. claude-1: 44.3% win rate

### 3. Response Length Impact
- Winner average: 1,569 characters
- Loser average: 1,293 characters
- Difference: 276 characters (21% longer)
- Statistical test: t = 17.76, p < 0.000001 (HIGHLY SIGNIFICANT)

### 4. Task Distribution
- Research/Information: Dominant category
- General Conversation: Second most common
- Coding: Significant but smaller segment
- Creative Writing, Math, Translation: Specialized use cases

### 5. Task-Specific Performance
- GPT-4 variants excel across most categories
- Claude competitive in creative writing
- Performance variation exists but GPT maintains lead

## Visualizations

All visualizations saved in `figures/` directory:

1. **winner_distribution.png** - User preference breakdown (35% model_a, 34% model_b, 31% tie)
2. **top_models.png** - Most frequently appearing models
3. **prompt_length_analysis.png** - Distribution of prompt characteristics
4. **model_win_rates.png** - Individual model performance rankings
5. **response_length_by_winner.png** - Response length comparison by outcome
6. **family_win_rates.png** - Model family performance comparison
7. **task_distribution.png** - Task category frequencies
8. **task_model_heatmap.png** - Task-specific model performance matrix
9. **final_data_overview.png** - Comprehensive dataset summary
10. **final_task_heatmap.png** - Detailed task-model performance

## Statistical Validation

All major findings achieved extremely high statistical significance:

| Test | Statistic | P-Value | Significance | Effect Size |
|------|-----------|---------|--------------|-------------|
| Chi-Square (Family vs Winner) | x^2 = 2777.40 | < 0.000001 | ★★★ | Very Large |
| T-Test (Response Length) | t = 17.76 | < 0.000001 | ★★★ | Medium-Large (d=0.31) |
| Correlation (Prompt vs Response) | r = 0.195 | < 0.001 | ★ | Weak |

**Legend:** ★★★ = Highly Significant, ★★ = Very Significant, ★ = Significant

## Limitations

- **Selection Bias:** Arena users may differ from general population
- **Temporal:** Data from 2023-2024 may not reflect newest models
- **Task Classification:** Keyword-based approach ~85% accurate
- **Causation:** Correlation demonstrated, not causation
- **Scope:** Text-only, English-language bias

## Future Work

- Extend to multi-modal models (image, code generation)
- Improve task classification with machine learning
- Temporal analysis of preference evolution
- Integrate actual model performance benchmarks
- Study demographic factors and user expertise
- Multi-turn conversation analysis

## Contributing

This is an academic project for DATA 720. While contributions are not currently accepted, feedback and suggestions are welcome via issues.

## Citation

If you use this analysis in your work, please cite:
```
Huang, A. (2025). LLM User Preference Analysis: A Data-Driven Study 
of Task-Specific Model Usage. DATA 720 Final Project. 
GitHub: https://github.com/AsaHuang-DS/LLM_Preference_Analysis
```

## Acknowledgments

- **LMSYS Organization** - Chatbot Arena platform and dataset
- **Hugging Face** - Dataset hosting infrastructure
- **MADS 720 Instructors** - Project guidance and feedback
- **Open Source Community** - Python data science ecosystem

## Contact

**Asa Huang**
- GitHub: [@AsaHuang-DS](https://github.com/AsaHuang-DS)
- Project: [LLM_Preference_Analysis](https://github.com/AsaHuang-DS/LLM_Preference_Analysis)

## Course Information

**Course:** DATA 720
**Institution:** University of North Carolina at Chapel Hill
**Semester:** Nov 2025
**Project Type:** Final Project - Data Analysis and Visualization

---

**Status:** ✅ Complete - All analysis finished, ready for presentation

**Last Updated:** Nov 2025
