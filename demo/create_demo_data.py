#!/usr/bin/env python3
"""
Create minimal demo data for testing the SV2000 frame monitoring system.

This script generates 5 synthetic articles with known frame patterns
to validate that the system works correctly.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path


def create_demo_articles():
    """Create 5 demo articles with known frame patterns."""
    
    articles = [
        {
            'article_id': 'demo_001',
            'title': 'Government Budget Cuts Spark Heated Debate',
            'content': '''The proposed government budget cuts have ignited fierce disagreement between political parties. Opposition leaders strongly criticized the administration's fiscal policies, calling them "economically destructive" and "morally irresponsible." The cuts are expected to reduce government spending by $2.5 billion over the next fiscal year, affecting social programs and infrastructure projects. Citizens expressed concern about the potential impact on their daily lives, with many families worried about reduced services. The debate reflects fundamental disagreements about the role of government in society and the best path forward for economic recovery.''',
            # Expected frames: High conflict, moderate economic, moderate responsibility
            'sv_conflict_q1_reflects_disagreement': 0.9,
            'sv_conflict_q2_disagreement_between_parties': 0.8,
            'sv_conflict_q3_conflict_between_groups': 0.7,
            'sv_conflict_q4_conflict_between_individuals': 0.3,
            'sv_human_q1_human_example_or_face': 0.4,
            'sv_human_q2_human_story_or_experience': 0.3,
            'sv_human_q3_emotional_angle': 0.5,
            'sv_human_q4_personal_vignettes': 0.2,
            'sv_human_q5_human_consequences': 0.6,
            'sv_econ_q1_financial_losses_gains': 0.8,
            'sv_econ_q2_costs_degree_expense': 0.9,
            'sv_econ_q3_economic_consequences': 0.7,
            'sv_moral_q1_moral_message': 0.4,
            'sv_moral_q2_social_prescriptions': 0.2,
            'sv_moral_q3_specific_social_groups': 0.3,
            'sv_resp_q1_government_ability_solve': 0.7,
            'sv_resp_q2_government_responsibility': 0.8,
            'sv_resp_q3_individual_responsibility': 0.3,
            'sv_resp_q4_problem_requires_action': 0.6,
            'sv_resp_q5_action_efficacy': 0.5
        },
        {
            'article_id': 'demo_002',
            'title': 'Local Mother Fights for Better School Funding',
            'content': '''Sarah Martinez, a single mother of two, has become the face of the campaign for increased school funding in her district. "My children deserve the same opportunities as kids in wealthier neighborhoods," she said, tears in her eyes as she spoke at the school board meeting. Martinez works two jobs to support her family but still finds time to advocate for educational equity. Her story resonates with many parents who feel their voices aren't being heard. The emotional testimony has galvanized community support, with dozens of families now joining the cause. Martinez's personal struggle highlights the human cost of underfunded education systems.''',
            # Expected frames: High human interest, low conflict, moderate responsibility
            'sv_conflict_q1_reflects_disagreement': 0.2,
            'sv_conflict_q2_disagreement_between_parties': 0.1,
            'sv_conflict_q3_conflict_between_groups': 0.3,
            'sv_conflict_q4_conflict_between_individuals': 0.1,
            'sv_human_q1_human_example_or_face': 0.9,
            'sv_human_q2_human_story_or_experience': 0.8,
            'sv_human_q3_emotional_angle': 0.9,
            'sv_human_q4_personal_vignettes': 0.7,
            'sv_human_q5_human_consequences': 0.8,
            'sv_econ_q1_financial_losses_gains': 0.3,
            'sv_econ_q2_costs_degree_expense': 0.4,
            'sv_econ_q3_economic_consequences': 0.2,
            'sv_moral_q1_moral_message': 0.3,
            'sv_moral_q2_social_prescriptions': 0.2,
            'sv_moral_q3_specific_social_groups': 0.4,
            'sv_resp_q1_government_ability_solve': 0.6,
            'sv_resp_q2_government_responsibility': 0.7,
            'sv_resp_q3_individual_responsibility': 0.5,
            'sv_resp_q4_problem_requires_action': 0.8,
            'sv_resp_q5_action_efficacy': 0.6
        },
        {
            'article_id': 'demo_003',
            'title': 'Tech Company Reports Record Quarterly Profits',
            'content': '''TechCorp announced record-breaking quarterly profits of $4.2 billion, representing a 35% increase from the same period last year. The company's stock price surged 12% following the earnings announcement, adding approximately $15 billion to its market capitalization. CEO John Smith attributed the success to strong demand for cloud services and artificial intelligence products. The financial results exceeded analyst expectations by a significant margin, with revenue reaching $28.7 billion for the quarter. Investors are optimistic about the company's future prospects, particularly in emerging technology markets. The strong performance is expected to drive continued investment in research and development.''',
            # Expected frames: High economic, low human interest, low conflict
            'sv_conflict_q1_reflects_disagreement': 0.1,
            'sv_conflict_q2_disagreement_between_parties': 0.0,
            'sv_conflict_q3_conflict_between_groups': 0.1,
            'sv_conflict_q4_conflict_between_individuals': 0.0,
            'sv_human_q1_human_example_or_face': 0.2,
            'sv_human_q2_human_story_or_experience': 0.1,
            'sv_human_q3_emotional_angle': 0.1,
            'sv_human_q4_personal_vignettes': 0.0,
            'sv_human_q5_human_consequences': 0.2,
            'sv_econ_q1_financial_losses_gains': 0.9,
            'sv_econ_q2_costs_degree_expense': 0.7,
            'sv_econ_q3_economic_consequences': 0.8,
            'sv_moral_q1_moral_message': 0.1,
            'sv_moral_q2_social_prescriptions': 0.0,
            'sv_moral_q3_specific_social_groups': 0.1,
            'sv_resp_q1_government_ability_solve': 0.2,
            'sv_resp_q2_government_responsibility': 0.1,
            'sv_resp_q3_individual_responsibility': 0.3,
            'sv_resp_q4_problem_requires_action': 0.2,
            'sv_resp_q5_action_efficacy': 0.3
        },
        {
            'article_id': 'demo_004',
            'title': 'Community Leaders Call for Ethical Business Practices',
            'content': '''Religious and community leaders gathered yesterday to call for higher ethical standards in business practices. "We have a moral obligation to ensure that our economic activities serve the greater good," said Reverend Michael Johnson at the interfaith gathering. The coalition emphasized the importance of corporate social responsibility and fair treatment of workers. They argued that businesses should be guided by principles of justice, compassion, and integrity rather than profit alone. The group plans to work with local companies to develop ethical guidelines and promote socially responsible business practices. "Our faith teaches us that we are our brother's keeper," added Rabbi Sarah Cohen, highlighting the moral imperative for ethical conduct in all aspects of life.''',
            # Expected frames: High moral, moderate responsibility, low economic
            'sv_conflict_q1_reflects_disagreement': 0.2,
            'sv_conflict_q2_disagreement_between_parties': 0.1,
            'sv_conflict_q3_conflict_between_groups': 0.2,
            'sv_conflict_q4_conflict_between_individuals': 0.1,
            'sv_human_q1_human_example_or_face': 0.4,
            'sv_human_q2_human_story_or_experience': 0.3,
            'sv_human_q3_emotional_angle': 0.3,
            'sv_human_q4_personal_vignettes': 0.2,
            'sv_human_q5_human_consequences': 0.4,
            'sv_econ_q1_financial_losses_gains': 0.2,
            'sv_econ_q2_costs_degree_expense': 0.1,
            'sv_econ_q3_economic_consequences': 0.3,
            'sv_moral_q1_moral_message': 0.9,
            'sv_moral_q2_social_prescriptions': 0.8,
            'sv_moral_q3_specific_social_groups': 0.7,
            'sv_resp_q1_government_ability_solve': 0.3,
            'sv_resp_q2_government_responsibility': 0.2,
            'sv_resp_q3_individual_responsibility': 0.7,
            'sv_resp_q4_problem_requires_action': 0.6,
            'sv_resp_q5_action_efficacy': 0.5
        },
        {
            'article_id': 'demo_005',
            'title': 'Climate Change Action Plan Faces Implementation Challenges',
            'content': '''The city's ambitious climate change action plan is encountering significant implementation challenges, according to a new report from the environmental committee. While the plan calls for a 50% reduction in carbon emissions by 2030, current progress suggests the city will fall short of this goal without immediate intervention. The report identifies funding shortfalls, regulatory barriers, and coordination problems as key obstacles. Environmental groups argue that the city government must take decisive action to address these challenges, while business leaders express concern about the economic impact of proposed regulations. The committee recommends establishing a dedicated climate action office with sufficient resources and authority to drive implementation. Citizens are calling for urgent action, emphasizing that the window for effective climate response is rapidly closing.''',
            # Expected frames: High responsibility, moderate conflict, moderate economic
            'sv_conflict_q1_reflects_disagreement': 0.5,
            'sv_conflict_q2_disagreement_between_parties': 0.4,
            'sv_conflict_q3_conflict_between_groups': 0.6,
            'sv_conflict_q4_conflict_between_individuals': 0.2,
            'sv_human_q1_human_example_or_face': 0.3,
            'sv_human_q2_human_story_or_experience': 0.2,
            'sv_human_q3_emotional_angle': 0.4,
            'sv_human_q4_personal_vignettes': 0.1,
            'sv_human_q5_human_consequences': 0.5,
            'sv_econ_q1_financial_losses_gains': 0.4,
            'sv_econ_q2_costs_degree_expense': 0.5,
            'sv_econ_q3_economic_consequences': 0.6,
            'sv_moral_q1_moral_message': 0.3,
            'sv_moral_q2_social_prescriptions': 0.2,
            'sv_moral_q3_specific_social_groups': 0.2,
            'sv_resp_q1_government_ability_solve': 0.8,
            'sv_resp_q2_government_responsibility': 0.9,
            'sv_resp_q3_individual_responsibility': 0.6,
            'sv_resp_q4_problem_requires_action': 0.9,
            'sv_resp_q5_action_efficacy': 0.7
        }
    ]
    
    return articles


def create_expected_outputs():
    """Create expected outputs for validation."""
    
    # Expected frame scores after processing (approximate)
    expected_frames = {
        'demo_001': {
            'conflict': 0.675,  # Mean of [0.9, 0.8, 0.7, 0.3]
            'human': 0.400,     # Mean of [0.4, 0.3, 0.5, 0.2, 0.6]
            'econ': 0.800,      # Mean of [0.8, 0.9, 0.7]
            'moral': 0.300,     # Mean of [0.4, 0.2, 0.3]
            'resp': 0.580       # Mean of [0.7, 0.8, 0.3, 0.6, 0.5]
        },
        'demo_002': {
            'conflict': 0.175,  # Low conflict
            'human': 0.820,     # High human interest
            'econ': 0.300,      # Low economic
            'moral': 0.300,     # Low moral
            'resp': 0.640       # Moderate responsibility
        },
        'demo_003': {
            'conflict': 0.050,  # Very low conflict
            'human': 0.120,     # Very low human interest
            'econ': 0.800,      # High economic
            'moral': 0.067,     # Very low moral
            'resp': 0.220       # Low responsibility
        },
        'demo_004': {
            'conflict': 0.150,  # Low conflict
            'human': 0.320,     # Low-moderate human interest
            'econ': 0.200,      # Low economic
            'moral': 0.800,     # High moral
            'resp': 0.460       # Moderate responsibility
        },
        'demo_005': {
            'conflict': 0.425,  # Moderate conflict
            'human': 0.300,     # Low-moderate human interest
            'econ': 0.500,      # Moderate economic
            'moral': 0.233,     # Low moral
            'resp': 0.780       # High responsibility
        }
    }
    
    # Expected binary labels (using threshold 0.1)
    expected_binary = {}
    for article_id, frames in expected_frames.items():
        expected_binary[article_id] = {
            frame: 1 if score >= 0.1 else 0 
            for frame, score in frames.items()
        }
    
    return expected_frames, expected_binary


def main():
    """Create demo data files."""
    
    # Create demo directory
    demo_dir = Path(__file__).parent
    demo_dir.mkdir(exist_ok=True)
    
    # Create articles
    articles = create_demo_articles()
    
    # Convert to DataFrame
    df = pd.DataFrame(articles)
    
    # Save as parquet (main format)
    df.to_parquet(demo_dir / "demo_data.parquet", index=False)
    print(f"Created: {demo_dir / 'demo_data.parquet'}")
    
    # Save as CSV (backup format)
    df.to_csv(demo_dir / "demo_data.csv", index=False)
    print(f"Created: {demo_dir / 'demo_data.csv'}")
    
    # Create train/valid/test splits (same data for demo)
    df.to_parquet(demo_dir / "train.parquet", index=False)
    df.to_parquet(demo_dir / "valid.parquet", index=False)
    df.to_parquet(demo_dir / "test.parquet", index=False)
    print(f"Created train/valid/test splits")
    
    # Create expected outputs
    expected_frames, expected_binary = create_expected_outputs()
    
    # Save expected outputs
    expected_data = {
        'regression_scores': expected_frames,
        'binary_labels': expected_binary,
        'metadata': {
            'num_articles': len(articles),
            'frame_names': ['conflict', 'human', 'econ', 'moral', 'resp'],
            'threshold': 0.1,
            'aggregation': 'mean',
            'normalization': 'none'
        }
    }
    
    with open(demo_dir / "expected_outputs.json", 'w') as f:
        json.dump(expected_data, f, indent=2)
    print(f"Created: {demo_dir / 'expected_outputs.json'}")
    
    # Create demo config
    demo_config = {
        'project': {
            'name': 'sv2000_demo',
            'seed': 42,
            'output_dir': 'demo/reports',
            'ckpt_dir': 'demo/checkpoints'
        },
        'data': {
            'train_path': 'demo/train.parquet',
            'valid_path': 'demo/valid.parquet',
            'test_path': 'demo/test.parquet',
            'text_fields': ['title', 'content'],
            'id_field': 'article_id',
            'frame_defs': {
                'conflict': [
                    'sv_conflict_q1_reflects_disagreement',
                    'sv_conflict_q2_disagreement_between_parties',
                    'sv_conflict_q3_conflict_between_groups',
                    'sv_conflict_q4_conflict_between_individuals'
                ],
                'human': [
                    'sv_human_q1_human_example_or_face',
                    'sv_human_q2_human_story_or_experience',
                    'sv_human_q3_emotional_angle',
                    'sv_human_q4_personal_vignettes',
                    'sv_human_q5_human_consequences'
                ],
                'econ': [
                    'sv_econ_q1_financial_losses_gains',
                    'sv_econ_q2_costs_degree_expense',
                    'sv_econ_q3_economic_consequences'
                ],
                'moral': [
                    'sv_moral_q1_moral_message',
                    'sv_moral_q2_social_prescriptions',
                    'sv_moral_q3_specific_social_groups'
                ],
                'resp': [
                    'sv_resp_q1_government_ability_solve',
                    'sv_resp_q2_government_responsibility',
                    'sv_resp_q3_individual_responsibility',
                    'sv_resp_q4_problem_requires_action',
                    'sv_resp_q5_action_efficacy'
                ]
            }
        },
        'label': {
            'regression_agg': 'mean',
            'normalize': 'none',
            'presence_threshold': 0.1
        },
        'model': {
            'backbone': 'longformer',
            'pretrained_name': 'allenai/longformer-base-4096',
            'max_length': 512,  # Shorter for demo
            'long_text_strategy': 'truncate_head_tail',
            'use_title': True
        },
        'training': {
            'epochs': 2,  # Very short for demo
            'batch_size': 2,
            'lr': 2e-5
        }
    }
    
    import yaml
    with open(demo_dir / "demo_config.yaml", 'w') as f:
        yaml.dump(demo_config, f, default_flow_style=False)
    print(f"Created: {demo_dir / 'demo_config.yaml'}")
    
    # Create validation script
    validation_script = '''#!/usr/bin/env python3
"""
Validate demo data processing.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import load_sv2000_dataframe, build_labels
import json

def main():
    # Load demo data
    df = load_sv2000_dataframe('demo/demo_data.parquet')
    print(f"Loaded {len(df)} demo articles")
    
    # Load expected outputs
    with open('demo/expected_outputs.json') as f:
        expected = json.load(f)
    
    # Build labels
    frame_defs = {
        'conflict': ['sv_conflict_q1_reflects_disagreement', 'sv_conflict_q2_disagreement_between_parties', 'sv_conflict_q3_conflict_between_groups', 'sv_conflict_q4_conflict_between_individuals'],
        'human': ['sv_human_q1_human_example_or_face', 'sv_human_q2_human_story_or_experience', 'sv_human_q3_emotional_angle', 'sv_human_q4_personal_vignettes', 'sv_human_q5_human_consequences'],
        'econ': ['sv_econ_q1_financial_losses_gains', 'sv_econ_q2_costs_degree_expense', 'sv_econ_q3_economic_consequences'],
        'moral': ['sv_moral_q1_moral_message', 'sv_moral_q2_social_prescriptions', 'sv_moral_q3_specific_social_groups'],
        'resp': ['sv_resp_q1_government_ability_solve', 'sv_resp_q2_government_responsibility', 'sv_resp_q3_individual_responsibility', 'sv_resp_q4_problem_requires_action', 'sv_resp_q5_action_efficacy']
    }
    
    y_reg, y_cls, frame_names = build_labels(
        df, frame_defs, 'mean', 'none', 0.1
    )
    
    print(f"Built labels: {y_reg.shape}, {y_cls.shape}")
    print(f"Frame names: {frame_names}")
    
    # Validate against expected
    tolerance = 0.05
    all_passed = True
    
    for i, article_id in enumerate(df['article_id']):
        expected_reg = expected['regression_scores'][article_id]
        for j, frame in enumerate(frame_names):
            actual = y_reg[i, j]
            expected_val = expected_reg[frame]
            if abs(actual - expected_val) > tolerance:
                print(f"FAIL: {article_id}.{frame}: expected {expected_val:.3f}, got {actual:.3f}")
                all_passed = False
            else:
                print(f"PASS: {article_id}.{frame}: {actual:.3f}")
    
    if all_passed:
        print("\\n✓ All validation tests passed!")
        return 0
    else:
        print("\\n✗ Some validation tests failed!")
        return 1

if __name__ == "__main__":
    exit(main())
'''
    
    with open(demo_dir / "validate_demo.py", 'w', encoding='utf-8') as f:
        f.write(validation_script)
    print(f"Created: {demo_dir / 'validate_demo.py'}")
    
    print(f"\nDemo data created successfully!")
    print(f"To validate: python demo/validate_demo.py")
    print(f"To run demo: python scripts/reproduce_training.py --config demo/demo_config.yaml --dry_run")


if __name__ == "__main__":
    main()