# SV2000 Label Construction Rules

## Overview

This document defines the **exact** and **reproducible** rules for constructing frame labels from the SV2000 questionnaire data. These rules are critical for ensuring consistent results across different experiments and implementations.

## SV2000 Framework Background

The SV2000 (Semetko & Valkenburg, 2000) framework identifies 5 news framing dimensions through 20 questionnaire items. Each item is typically scored on a scale (e.g., 0-1 or 1-5), and multiple items contribute to each frame dimension.

## Frame Definitions and Item Mappings

### 1. Conflict Frame (4 items)
**Theoretical Definition**: Emphasizes disagreement and conflict between parties, groups, or individuals.

**SV2000 Items**:
- `sv_conflict_q1_reflects_disagreement`: Does the story reflect disagreement?
- `sv_conflict_q2_disagreement_between_parties`: Does the story refer to disagreement between parties?
- `sv_conflict_q3_conflict_between_groups`: Does the story refer to conflict between groups?
- `sv_conflict_q4_conflict_between_individuals`: Does the story refer to conflict between individuals?

### 2. Human Interest Frame (5 items)
**Theoretical Definition**: Brings a human face or emotional angle to the presentation of an event, issue, or problem.

**SV2000 Items**:
- `sv_human_q1_human_example_or_face`: Does the story provide a human example or human face?
- `sv_human_q2_human_story_or_experience`: Does the story employ adjectives or personal vignettes?
- `sv_human_q3_emotional_angle`: Does the story emphasize how individuals are affected?
- `sv_human_q4_personal_vignettes`: Does the story go into the private or personal lives?
- `sv_human_q5_human_consequences`: Does the story contain visual information that might generate feelings?

### 3. Economic Frame (3 items)
**Theoretical Definition**: Reports an event, problem, or issue in terms of economic consequences.

**SV2000 Items**:
- `sv_econ_q1_financial_losses_gains`: Does the story mention financial losses or gains?
- `sv_econ_q2_costs_degree_expense`: Does the story mention the costs/degree of expense involved?
- `sv_econ_q3_economic_consequences`: Does the story refer to economic consequences?

### 4. Moral Frame (3 items)
**Theoretical Definition**: Puts the event, problem, or issue in the context of religious tenets or moral messages.

**SV2000 Items**:
- `sv_moral_q1_moral_message`: Does the story contain moral message?
- `sv_moral_q2_social_prescriptions`: Does the story make reference to morality, God, and other religious tenets?
- `sv_moral_q3_specific_social_groups`: Does the story offer specific social prescriptions about how to behave?

### 5. Responsibility Frame (5 items)
**Theoretical Definition**: Presents an issue or problem in such a way as to attribute responsibility for its cause or solution.

**SV2000 Items**:
- `sv_resp_q1_government_ability_solve`: Does the story suggest that some level of government has the ability to alleviate the problem?
- `sv_resp_q2_government_responsibility`: Does the story suggest that some level of government is responsible for the issue/problem?
- `sv_resp_q3_individual_responsibility`: Does the story suggest solution(s) to the problem/issue?
- `sv_resp_q4_problem_requires_action`: Does the story suggest that an individual (or group of people in society) is responsible?
- `sv_resp_q5_action_efficacy`: Does the story suggest the problem requires urgent action?

## Label Construction Pipeline

### Step 1: Item-Level Validation
1. **Missing Value Handling**: 
   - Strategy: `zero_fill` (replace NaN with 0.0)
   - Alternative: `skip_sample` (exclude samples with too many missing values)
   - Minimum valid items per frame: 1

2. **Range Validation**:
   - Expected input range: [0.0, 1.0] or [1.0, 5.0] (auto-detected)
   - Out-of-range values are clipped to valid range

### Step 2: Frame-Level Aggregation
For each frame, combine multiple items into a single score:

```python
# Example for Conflict frame
conflict_items = [
    "sv_conflict_q1_reflects_disagreement",
    "sv_conflict_q2_disagreement_between_parties", 
    "sv_conflict_q3_conflict_between_groups",
    "sv_conflict_q4_conflict_between_individuals"
]

# Aggregation methods:
if regression_agg == "mean":
    conflict_score = np.nanmean(df[conflict_items], axis=1)
elif regression_agg == "weighted_mean":
    # Equal weights for now, can be customized per frame
    weights = [1.0, 1.0, 1.0, 1.0]  
    conflict_score = np.average(df[conflict_items], weights=weights, axis=1)
```

### Step 3: Normalization
Apply normalization to frame-level scores:

```python
if normalize == "minmax_0_1":
    # Scale to [0, 1] range
    score_min, score_max = scores.min(), scores.max()
    if score_max > score_min:
        scores = (scores - score_min) / (score_max - score_min)
        
elif normalize == "zscore":
    # Standardize to mean=0, std=1
    scores = (scores - scores.mean()) / scores.std()
    
elif normalize == "none":
    # No normalization
    pass
```

### Step 4: Binary Classification Labels
Convert regression scores to binary presence labels:

```python
# Threshold-based conversion
presence_labels = (regression_scores >= presence_threshold).astype(int)

# Default threshold: 0.1 (conservative, captures weak presence)
# Alternative thresholds: 0.3 (moderate), 0.5 (strong presence)
```

## Output Format

The label construction process produces:

1. **Regression Labels** (`y_reg`): Shape [N, 5], dtype float32
   - Continuous scores for frame intensity
   - Range: [0, 1] after normalization
   - Used for regression loss and Pearson correlation metrics

2. **Classification Labels** (`y_cls`): Shape [N, 5], dtype int64  
   - Binary labels for frame presence/absence
   - Values: 0 (absent) or 1 (present)
   - Used for classification loss and precision/recall metrics

3. **Frame Names**: List of 5 strings
   - Order: ["conflict", "human", "econ", "moral", "resp"]
   - Consistent ordering across all outputs

## Configuration Parameters

All label construction parameters are specified in the config file:

```yaml
label:
  regression_agg: "mean"              # mean|weighted_mean|median
  normalize: "minmax_0_1"             # none|minmax_0_1|zscore  
  presence_threshold: 0.1             # Threshold for binary conversion
  min_valid_items_per_frame: 1        # Minimum non-NaN items required
  handle_missing_strategy: "zero_fill" # zero_fill|skip_sample
  validate_score_ranges: true         # Enable range validation
  expected_score_range: [0.0, 1.0]    # Expected input range
```

## Quality Control and Validation

### Automated Checks
1. **Item Coverage**: Verify all expected SV2000 columns are present
2. **Missing Data**: Report percentage of missing values per item
3. **Score Distribution**: Log min/max/mean/std for each frame
4. **Label Balance**: Report positive/negative class ratios
5. **Correlation Matrix**: Check inter-frame correlations

### Reproducibility Guarantees
1. **Deterministic Processing**: Fixed random seeds, consistent ordering
2. **Version Control**: All parameters logged in config and output files
3. **Validation Checksums**: MD5 hashes of processed labels for verification

## Usage Example

```python
from data.dataset import build_labels

# Load configuration
config = load_config("configs/longformer_sv2000.yaml")

# Build labels
y_reg, y_cls, frame_names = build_labels(
    df=train_df,
    frame_defs=config['data']['frame_defs'],
    regression_agg=config['label']['regression_agg'],
    normalize=config['label']['normalize'], 
    presence_threshold=config['label']['presence_threshold']
)

print(f"Regression labels shape: {y_reg.shape}")
print(f"Classification labels shape: {y_cls.shape}")
print(f"Frame names: {frame_names}")
```

## Troubleshooting

### Common Issues
1. **Missing Columns**: Check that SV2000 column names match exactly
2. **All-Zero Frames**: May indicate missing data or incorrect column mapping
3. **Extreme Imbalance**: Consider adjusting `presence_threshold`
4. **NaN Outputs**: Check `handle_missing_strategy` and input data quality

### Validation Commands
```bash
# Validate label construction
python -c "
from data.dataset import build_labels, load_sv2000_dataframe
df = load_sv2000_dataframe('data/train.parquet')
# ... run build_labels and print statistics
"
```