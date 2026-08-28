# Human Audit Directory Guide

This directory contains everything related to the **Human Audit** phase of the project. The purpose of this audit was to have humans manually label 200 model failures (as Visual, Language-Prior, Alignment, etc.) to see if human intuition matches the algorithm's predicted uncertainty source.

Below is a plain English guide to what every folder and file in this directory does.

## Folders
*   **`annotations/`** 
    Contains the raw, completed CSV files returned by the 3 independent human annotators.
*   **`images/`** 
    Contains the raw image files (like `.png` or `.jpg`) for the 200 samples that were audited, so the annotators could look at them.

## Data & Results (CSV Files)
*   **`blank_annotator_sheet.csv`**
    The blank template given to annotators to fill out. It contains the image filename, the question, the model's answer, and the gold (correct) answer, but hides all of the algorithm's scores so annotators aren't biased.
*   **`master_analysis_sheet.csv`**
    A "behind-the-scenes" spreadsheet that contains all the algorithm's raw data (scalar confidence, raw V/L/A scores, predicted dominant source) for those same 200 samples.
*   **`manual_adjudication_sheet.csv`**
    A spreadsheet containing only the 36 cases where all 3 annotators completely disagreed. This was generated so a final decision-maker (the adjudicator) could manually break the ties.
*   **`adjudicated_audit_results.csv`**
    The **final, complete dataset**. It perfectly merges the model's scores, the 3 human annotations, the majority votes, and the manual tie-breakers into one ultimate "Ground Truth" spreadsheet used for the final paper metrics.

## Documentation (Text/Markdown Files)
*   **`annotation_guide.md` / `annotation_guide.txt`**
    The official rulebook given to the annotators. It explains what V, L, A, MIXED, and UNCLEAR mean, and gives them examples of how to classify different types of failures.
*   **`sampling_plan.md`**
    Documentation explaining the math and logic behind *how* we picked the 200 samples out of the thousands of experiments we ran (e.g., ensuring we picked exactly 50 from each dataset).

## Python Scripts (.py Files)
*   **`run_audit_analysis.py`**
    The most important script here. It reads the annotations, calculates how well the annotators agreed with each other (Fleiss' Kappa), handles the voting logic, and prints out the final Model-vs-Human accuracy matrix for the paper.
*   **`generate_adjudication_sheet.py`**
    The script that scanned the annotations, found the 3-way ties, and created the `manual_adjudication_sheet.csv` for you to fill out.
*   **`sample_audit_cases.py`**
    The original script used at the very beginning to randomly pull the 200 samples from the massive experiment results folders to create this audit batch.
*   **`compute_wallclock.py`**
    A handy utility script that scans all the `.log` files in the whole repository to calculate exactly how many days/hours the experiments ran on the GPU. (Used for the Computational Cost section of the paper).
*   **`check_data.py`** & **`check_dupes.py`**
    Small debugging scripts used to make sure the CSV files weren't corrupted and didn't have duplicate rows.

## Other
*   **`images.zip`**
    Just a compressed version of the `images/` folder so it could be easily emailed or sent to the annotators.
