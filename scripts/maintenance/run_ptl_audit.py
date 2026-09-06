"""Run PTL feature audit and save results"""
import sys
import traceback

try:
    import pandas as pd

    sys.path.insert(0, 'src')
    from transportability.ptl import build_feature_audit

    # Load a real dataframe to analyze
    frame = pd.read_parquet('results/tables/ptl_examples.parquet')
    print(f"Loaded {len(frame)} rows, {len(frame.columns)} columns")

    # Run feature audit in deployment mode
    audit_df = build_feature_audit(frame, feature_mode='deployment')
    print(f"Audit created: {len(audit_df)} features")
    print(f"Columns: {list(audit_df.columns)}")

    # Count categories - use correct column name
    dep_count = len(audit_df[audit_df['deployment_available'] == True])
    eval_count = len(audit_df[audit_df['evaluation_only'] == True])
    target_count = len(audit_df[audit_df['target_or_label'] == True])
    excluded_count = len(audit_df[audit_df['excluded_from_ptl'] == True])

    print(f'Deployment available: {dep_count}')
    print(f'Evaluation only: {eval_count}')
    print(f'Target/label: {target_count}')
    print(f'Excluded from PTL: {excluded_count}')

    # Save to file
    audit_df.to_csv('results/tables/ptl_feature_audit.csv', index=False)
    print('Saved to results/tables/ptl_feature_audit.csv')
    print('DONE')

except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)