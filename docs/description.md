# Overview

Profiles how clonotype or cluster abundance distributes across categorical metadata dimensions such as tissue compartments, longitudinal timepoints, and biological subjects. Designed for in vivo antibody and TCR discovery campaigns where understanding clonal dynamics across experimental conditions is essential.

## Grouping analysis

Computes per-clone restriction index (how concentrated a clone is across groups), dominant group assignment, and breadth (number of groups where present). In population-level mode with a subject variable, produces consensus metrics averaged across subjects.

## Temporal analysis

Tracks frequency trajectories over ordered timepoints. Computes Log2 Peak Delta (expansion from first detection to peak), Temporal Shift Index (weighted average timepoint), and Log2 Kinetic Delta (last-to-first change). Selects top expanding clones for trajectory visualization.

## Cross-subject convergence

When a subject variable is provided, computes subject prevalence (how many subjects share each clone) and prevalence fraction. In population-level mode, averages per-subject metrics to identify clones with consistent behavior across the cohort.

## Normalization

Supports relative frequency normalization and centered log-ratio (CLR) transform for compositional data. CLR is applied globally in population-level mode and per-subject in intra-subject mode, with multiplicative replacement for zero values.

## Outputs

- Summary table with all computed metrics per clone
- Distribution heatmap showing clone distribution across categories
- Temporal trajectory plot for top expanding clones
- Subject prevalence histogram
