# LLM User Preference Analysis: A Data-Driven Study of Task-Specific Model Usage

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

## Overview

This project conducts a comprehensive analysis of user preferences across Large Language Models (LLMs) using real-world conversation data and model performance metrics. By integrating user behavior data with technical model characteristics, this study identifies task-specific usage patterns and explores the relationship between model attributes (speed, cost, performance) and user satisfaction.

**Key Question:** Do users demonstrate statistically significant preferences for specific LLMs based on task type, and how do model characteristics influence these preferences?

## Project Description

As artificial intelligence tools become increasingly integrated into daily workflows, understanding how users interact with different LLMs across various tasks is critical. This project analyzes over 55,000 real user conversations spanning 70+ state-of-the-art language models to uncover:

- Task-specific model preferences (coding, creative writing, data analysis, research)
- Correlations between model performance metrics and user satisfaction
- Statistical significance of preference patterns across different use cases
- Insights for model selection optimization and product development

The analysis employs descriptive statistics, hypothesis testing, and data visualization techniques to deliver actionable insights for AI developers, end users, and researchers studying human-AI interaction.

## Research Framework

### Primary Hypothesis
Users exhibit statistically significant preferences for certain LLMs when performing specific tasks (e.g., coding, creative writing, data analysis). Furthermore, model characteristics such as response speed, operational cost, and benchmark performance metrics correlate with user preference patterns.

### Research Questions
1. Which LLMs are most frequently preferred by users for coding-related tasks compared to creative or conversational tasks?
2. Do model performance characteristics (speed, cost, benchmark scores) correlate with user preference rates?
3. What task categories dominate user interactions with LLMs, and how does this vary across different models?
4. Are there statistically significant differences in user satisfaction across LLM models when controlling for task type?

### Significance
This research contributes to:
- **Product Development:** Informing targeted model optimization and feature prioritization
- **User Experience:** Guiding users toward optimal model selection for specific tasks
- **Academic Research:** Advancing understanding of human-AI interaction patterns
- **Market Analysis:** Providing insights into competitive positioning in the LLM landscape

## Datasets

### Primary Dataset: Chatbot Arena Human Preferences
- **Source:** [Hugging Face - lmsys/chatbot_arena_conversations](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k)
- **Size:** 55,000+ real-world user conversations
- **Models Covered:** 70+ LLMs including GPT-4, Claude 2, Llama 2, Gemini, Mistral, and others
- **Features:** User prompts, model responses, preference votes, timestamps, metadata
- **Collection:** Ongoing data from LMSYS Chatbot Arena platform

### Secondary Dataset: Large Language Models Comparison
- **Source:** [Kaggle - Large Language Models Comparison Dataset](https://www.kaggle.com/datasets/samayashar/large-language-models-comparison-dataset)
- **Content:** Model specifications, performance benchmarks, speed metrics, cost data
- **Purpose:** Enriches user preference analysis with technical model characteristics
- **Integration:** Merged with primary dataset on model names for comprehensive analysis

**Note:** Due to file size constraints (>100MB), datasets are not included in this repository. See [Data Download Instructions](#data-download-instructions) below.

## Project Structure

```
llm-preference-analysis/
│
├── data/
│   ├── raw/                          # Original datasets (not tracked by git)
│   │   ├── chatbot_arena.csv         # Hugging Face dataset
│   │   └── llm_comparison.csv        # Kaggle dataset
│   └── processed/                    # Cleaned and merged data (not tracked)
│
├── notebooks/
│   ├── 00_data_download.ipynb        # Dataset acquisition and initial setup
│   ├── 01_data_exploration.ipynb     # Initial data profiling and quality assessment
│   ├── 02_data_cleaning.ipynb        # Preprocessing and feature engineering
│   ├── 03_exploratory_analysis.ipynb # Comprehensive EDA with visualizations
│   ├── 04_statistical_analysis.ipynb # Hypothesis testing and deep dive
│   └── 05_final_visualizations.ipynb # Publication-quality figures
│
├── figures/                          # Saved plots and visualizations
│
├── src/                              # Helper functions and utilities
│   └── utils.py
│
├── .gitignore                        # Git ignore rules
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
└── LICENSE                           # MIT License

```

## Installation and Setup

### Prerequisites
- Python 3.10 or higher
- Anaconda or Miniconda
- Git
- Kaggle account (for dataset download)

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

### Data Download Instructions

**Important:** Data files are not included in this repository due to size constraints.

Run the data download notebook:
```bash
jupyter notebook notebooks/00_data_download.ipynb
```

This notebook will:
1. Download Chatbot Arena dataset from Hugging Face (~175 MB)
2. Download LLM comparison dataset from Kaggle
3. Save both datasets to `data/raw/`
4. Perform initial data validation

**Expected download time:** 5-10 minutes depending on internet speed.

## Methodology

### Analytical Approach

**Phase 1: Data Acquisition and Preparation**
- Load datasets from Hugging Face and Kaggle
- Initial data profiling and quality assessment
- Merge datasets on model names

**Phase 2: Data Cleaning and Feature Engineering**
- Handle missing values and inconsistencies
- Text preprocessing and standardization
- Create derived features: conversation length, task categories, win rates

**Phase 3: Exploratory Data Analysis**
- Univariate analysis: distributions of key variables
- Bivariate analysis: relationships between model characteristics and preferences
- Visualization: 15+ charts including bar plots, heatmaps, box plots, scatter plots

**Phase 4: Statistical Analysis**
- Task classification using keyword-based approach
- Hypothesis testing: chi-square, t-tests, ANOVA
- Correlation analysis between model metrics and user preferences
- Comparative analysis across models and task types

**Phase 5: Insights and Reporting**
- Synthesis of key findings
- Publication-quality visualizations
- Written report and presentation materials

### Technical Stack

**Core Libraries:**
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `scipy` - Statistical functions and hypothesis testing

**Visualization:**
- `matplotlib` - Foundational plotting
- `seaborn` - Statistical visualizations

**Data Loading:**
- `datasets` (Hugging Face) - Load Chatbot Arena data
- `kaggle` - Download Kaggle datasets

**Development:**
- `jupyter` - Interactive notebooks
- `git` - Version control

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
1. `00_data_download.ipynb` - Download and verify datasets
2. `01_data_exploration.ipynb` - Initial data exploration
3. `02_data_cleaning.ipynb` - Data preprocessing
4. `03_exploratory_analysis.ipynb` - Comprehensive EDA
5. `04_statistical_analysis.ipynb` - Hypothesis testing
6. `05_final_visualizations.ipynb` - Create final figures

### Notebook Contents

Each notebook includes:
- **Research Context:** Relevant hypothesis and research questions
- **Code with Explanations:** Detailed comments explaining logic and methodology
- **Visualizations:** Clear, labeled plots with interpretations
- **Key Findings:** Summary of insights from each analysis stage
- **Next Steps:** Preview of subsequent analysis phases

## Key Findings

*This section will be updated as analysis progresses.*

### Preliminary Insights
- Dataset contains 55,000+ conversations across 70+ LLM models
- Distribution of conversations by model, task type, and user preferences
- Initial patterns in task-specific model usage

### Statistical Results
- [To be completed after analysis]

### Visualizations
See `figures/` directory for all generated visualizations.

## Project Timeline

- **Weeks 1-9:** Problem identification and dataset selection
- **Week 9:** Project proposal submission
- **Weeks 9-10:** Scope refinement with instructor feedback
- **Weeks 10-12:** Data analysis execution
- **Week 14:** Digital poster presentation
- **End of Semester:** Final report and code submission

## Contributing

This is an academic project for MADS 720. While contributions are not currently accepted, feedback and suggestions are welcome via issues.

## Citation

If you use this analysis or methodology in your work, please cite:

```
Huang, A. (2024). LLM User Preference Analysis: A Data-Driven Study of Task-Specific Model Usage.
MADS 720 Final Project. GitHub: https://github.com/AsaHuang-DS/LLM_Preference_Analysis
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **LMSYS Organization** for maintaining the Chatbot Arena platform and dataset
- **Kaggle Community** for the LLM comparison dataset
- **MADS 720 Instructors** for project guidance and feedback
- **Hugging Face** for dataset hosting infrastructure

## Contact

**Asa Huang**
- GitHub: [@AsaHuang-DS](https://github.com/AsaHuang-DS)
- Project Link: [https://github.com/AsaHuang-DS/LLM_Preference_Analysis](https://github.com/AsaHuang-DS/LLM_Preference_Analysis)

## Course Information

**Course:** MADS 720 - Data Science  
**Institution:** [Your Institution]  
**Semester:** Fall 2024  
**Project Type:** Final Project - Data Analysis and Visualization

---

**Last Updated:** October 2025
**Status:** In Progress - Data Exploration Phase
