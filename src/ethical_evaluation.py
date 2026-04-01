"""
Ethical & Societal Evaluation Module
=====================================
This module provides comprehensive ethical analysis and documentation
for the Predictive Policing Decision Support System.

Key Ethical Considerations:
1. Reporting Bias Impact on ML
2. Risks of Predictive Policing
3. Fairness and Equity
4. Transparency and Explainability
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class EthicalEvaluator:
    """
    Evaluates ethical implications of crime prediction models
    """
    
    def __init__(self):
        self.ethical_report = []
        
    def analyze_reporting_bias(self, df, crime_col='TOTAL IPC CRIMES'):
        """
        Analyze potential reporting bias in the dataset
        
        Reporting bias occurs when:
        - Some states/districts over-report crimes
        - Some states/districts under-report crimes
        - Reporting practices vary over time
        """
        print("\n" + "=" * 60)
        print("📊 REPORTING BIAS ANALYSIS")
        print("=" * 60)
        
        # Aggregate to state level
        state_df = df.groupby('STATE/UT')[crime_col].agg(['sum', 'mean', 'std', 'count'])
        state_df['cv'] = (state_df['std'] / state_df['mean']) * 100
        
        findings = []
        
        # 1. High variance within states suggests inconsistent reporting
        high_variance_states = state_df[state_df['cv'] > 100]
        if len(high_variance_states) > 0:
            findings.append({
                'type': 'High Intra-state Variance',
                'description': 'States with CV > 100% have highly inconsistent district-level reporting',
                'affected': list(high_variance_states.index),
                'risk': 'HIGH',
                'mitigation': 'Use state-level aggregation instead of district predictions'
            })
        
        # 2. Check for suspiciously low crime rates
        very_low = state_df[state_df['mean'] < state_df['mean'].quantile(0.1)]
        if len(very_low) > 0:
            findings.append({
                'type': 'Suspiciously Low Rates',
                'description': 'States with very low mean crime may indicate under-reporting',
                'affected': list(very_low.index),
                'risk': 'MEDIUM',
                'mitigation': 'Do not interpret low predictions as "safe" areas'
            })
        
        # 3. Check for outlier states
        z_scores = (state_df['sum'] - state_df['sum'].mean()) / state_df['sum'].std()
        outliers = state_df[np.abs(z_scores) > 2]
        if len(outliers) > 0:
            findings.append({
                'type': 'Statistical Outliers',
                'description': 'States with extreme values may skew model predictions',
                'affected': list(outliers.index),
                'risk': 'MEDIUM',
                'mitigation': 'Consider separate models for outlier states'
            })
        
        # Print findings
        for i, finding in enumerate(findings, 1):
            print(f"\n⚠️ Finding {i}: {finding['type']}")
            print(f"   Description: {finding['description']}")
            print(f"   Risk Level: {finding['risk']}")
            print(f"   Affected: {', '.join(finding['affected'][:5])}{'...' if len(finding['affected']) > 5 else ''}")
            print(f"   Mitigation: {finding['mitigation']}")
        
        self.ethical_report.extend(findings)
        return findings
    
    def generate_ethical_guidelines(self):
        """
        Generate ethical guidelines for using the prediction system
        """
        guidelines = """
╔══════════════════════════════════════════════════════════════╗
║           ETHICAL GUIDELINES FOR SYSTEM USE                  ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. PURPOSE LIMITATION                                       ║
║     ✓ Use ONLY for resource allocation planning              ║
║     ✓ Use for identifying areas needing social services      ║
║     ✗ DO NOT use for targeting individuals                   ║
║     ✗ DO NOT use for discriminatory practices                ║
║                                                              ║
║  2. INTERPRETATION GUIDELINES                                ║
║     ✓ HIGH risk = needs more community support               ║
║     ✓ LOW risk = may indicate under-reporting                ║
║     ✗ DO NOT interpret as "dangerous" vs "safe" areas        ║
║     ✗ DO NOT use for profiling communities                   ║
║                                                              ║
║  3. DATA LIMITATIONS                                         ║
║     ✓ Data reflects REPORTED crime only                      ║
║     ✓ Reporting rates vary by region and demographics        ║
║     ✓ Missing: population data, socioeconomic factors        ║
║     ✓ No individual-level data is used                       ║
║                                                              ║
║  4. TRANSPARENCY REQUIREMENTS                                ║
║     ✓ All predictions must include confidence levels         ║
║     ✓ Feature importance must be disclosed                   ║
║     ✓ Model limitations must accompany all outputs           ║
║     ✓ Regular bias audits are required                       ║
║                                                              ║
║  5. ACCOUNTABILITY                                           ║
║     ✓ Human review required before any action                ║
║     ✓ Appeals process for affected communities               ║
║     ✓ Regular model retraining with updated data             ║
║     ✓ External audit trails maintained                       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
        print(guidelines)
        return guidelines
    
    def analyze_predictive_policing_risks(self):
        """
        Document risks associated with predictive policing
        """
        risks = """
╔══════════════════════════════════════════════════════════════╗
║         RISKS OF PREDICTIVE POLICING SYSTEMS                 ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  🔴 HIGH RISK: Feedback Loops                                ║
║     Problem: More policing → more arrests → "higher crime"   ║
║     → more policing (self-fulfilling prophecy)               ║
║     Mitigation: Use crime REPORTS, not arrests; regular      ║
║     bias audits                                              ║
║                                                              ║
║  🔴 HIGH RISK: Discrimination Amplification                  ║
║     Problem: Historical bias in data gets encoded in ML      ║
║     Mitigation: State-level only; no demographic features;   ║
║     balanced class weights                                   ║
║                                                              ║
║  🟡 MEDIUM RISK: Over-reliance on Predictions                ║
║     Problem: Decision-makers may trust ML over judgment      ║
║     Mitigation: Mandatory human review; confidence scores;   ║
║     clear uncertainty communication                          ║
║                                                              ║
║  🟡 MEDIUM RISK: Privacy Concerns                            ║
║     Problem: Detailed predictions could identify individuals ║
║     Mitigation: Aggregate to state level; no individual      ║
║     predictions; no demographic profiling                    ║
║                                                              ║
║  🟢 LOW RISK (in this system): Individual Targeting          ║
║     Status: MITIGATED by design                              ║
║     How: Only state/district level predictions; no personal  ║
║     data; focus on resource allocation                       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
        print(risks)
        return risks
    
    def justify_state_level_analysis(self):
        """
        Provide justification for state-level analysis approach
        """
        justification = """
╔══════════════════════════════════════════════════════════════╗
║      JUSTIFICATION FOR STATE-LEVEL ANALYSIS                  ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  WHY STATE-LEVEL (not individual/neighborhood)?              ║
║                                                              ║
║  1. ETHICAL PROTECTION                                       ║
║     • Prevents individual profiling                          ║
║     • Reduces discrimination risk                            ║
║     • Cannot be used for stop-and-frisk policies             ║
║                                                              ║
║  2. DATA QUALITY                                             ║
║     • Aggregation smooths reporting inconsistencies          ║
║     • Reduces impact of local data errors                    ║
║     • More stable statistical estimates                      ║
║                                                              ║
║  3. APPROPRIATE USE CASE                                     ║
║     • Budget allocation across states                        ║
║     • Policy planning at government level                    ║
║     • Social program targeting by region                     ║
║                                                              ║
║  4. LEGAL COMPLIANCE                                         ║
║     • Aligns with privacy regulations                        ║
║     • No personally identifiable information                 ║
║     • Transparent and auditable                              ║
║                                                              ║
║  WHAT THIS SYSTEM CANNOT DO:                                 ║
║  ✗ Predict crime for specific neighborhoods                  ║
║  ✗ Identify "high-risk" individuals                          ║
║  ✗ Guide patrol routes                                       ║
║  ✗ Support arrest decisions                                  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
        print(justification)
        return justification
    
    def generate_full_report(self, df):
        """
        Generate comprehensive ethical evaluation report
        """
        print("\n" + "=" * 60)
        print("📋 COMPREHENSIVE ETHICAL EVALUATION REPORT")
        print("   Predictive Policing Decision Support System")
        print("=" * 60)
        
        # 1. Reporting bias analysis
        self.analyze_reporting_bias(df)
        
        # 2. Risks
        self.analyze_predictive_policing_risks()
        
        # 3. Justification
        self.justify_state_level_analysis()
        
        # 4. Guidelines
        self.generate_ethical_guidelines()
        
        print("\n" + "=" * 60)
        print("✅ Ethical Evaluation Complete")
        print("   This report should accompany all model deployments")
        print("=" * 60)


def main():
    """Run ethical evaluation"""
    # Load data
    df = pd.read_csv("data/raw/dstrIPC_2013.csv")
    
    # Initialize evaluator
    evaluator = EthicalEvaluator()
    
    # Generate full report
    evaluator.generate_full_report(df)


if __name__ == "__main__":
    main()
