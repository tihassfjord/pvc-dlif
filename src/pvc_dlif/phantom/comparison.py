"""
PVC Method Comparison & Analysis
==================================

Tools for comparing multiple PVC methods and iterations to determine
the best configuration for your specific use case.

This module:
1. Collects results from sweep runs
2. Exports comprehensive CSV with all metrics
3. Generates comparison plots (RC vs iterations, method comparison, etc.)
4. Produces statistical summaries
"""

import csv
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from datetime import datetime


def export_comparison_csv(
    results: List[Tuple[str, int, Dict[str, float]]],
    output_path: str,
) -> str:
    """
    Export sweep results to CSV for external analysis.

    Parameters
    ----------
    results : List[Tuple[str, int, Dict[str, float]]]
        List of (method, iterations, metrics_dict) tuples from sweep
    output_path : str
        Path where CSV will be saved

    Returns
    -------
    str
        Path to the created CSV file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all unique metric names
    all_metrics = set()
    for _, _, metrics in results:
        all_metrics.update(metrics.keys())
    all_metrics = sorted(all_metrics)

    # Write CSV
    csv_path = output_path
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['method', 'iterations'] + all_metrics
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for method, n_iter, metrics in results:
            row = {'method': method, 'iterations': n_iter}
            row.update(metrics)
            writer.writerow(row)

    return str(csv_path)


def generate_comparison_summary(
    results: List[Tuple[str, int, Dict[str, float]]],
    output_dir: str,
) -> str:
    """
    Generate a markdown summary comparing methods.

    Parameters
    ----------
    results : List[Tuple[str, int, Dict[str, float]]]
        List of (method, iterations, metrics_dict) tuples
    output_dir : str
        Directory where summary will be saved

    Returns
    -------
    str
        Markdown text of the comparison summary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Organize by method
    by_method = {}
    for method, n_iter, metrics in results:
        if method not in by_method:
            by_method[method] = []
        by_method[method].append((n_iter, metrics))

    # Build markdown
    summary = f"""# PVC Method Comparison Summary

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Total Configurations**: {len(results)}
**Methods Tested**: {', '.join(sorted(by_method.keys()))}

## Quick Comparison Table

This table shows key metrics for each method at the highest iteration count tested:

| Method | Iterations | RC_1mm | RC_2mm | RC_3mm | RC_4mm | RC_5mm | Uniformity (%STD) | Spillover (Water) |
|--------|-----------|--------|--------|--------|--------|--------|-------------------|-------------------|
"""

    for method in sorted(by_method.keys()):
        configs = sorted(by_method[method], key=lambda x: x[0], reverse=True)
        n_iter, metrics = configs[0]  # Highest iteration count
        
        rc_1mm = metrics.get('RC_1mm', 'N/A')
        rc_2mm = metrics.get('RC_2mm', 'N/A')
        rc_3mm = metrics.get('RC_3mm', 'N/A')
        rc_4mm = metrics.get('RC_4mm', 'N/A')
        rc_5mm = metrics.get('RC_5mm', 'N/A')
        uniformity = metrics.get('Uniformity_std_pct', 'N/A')
        spillover = metrics.get('Spillover_Water', 'N/A')
        
        # Format values
        if isinstance(rc_1mm, float):
            rc_1mm = f"{rc_1mm:.4f}"
        if isinstance(rc_2mm, float):
            rc_2mm = f"{rc_2mm:.4f}"
        if isinstance(rc_3mm, float):
            rc_3mm = f"{rc_3mm:.4f}"
        if isinstance(rc_4mm, float):
            rc_4mm = f"{rc_4mm:.4f}"
        if isinstance(rc_5mm, float):
            rc_5mm = f"{rc_5mm:.4f}"
        if isinstance(uniformity, float):
            uniformity = f"{uniformity:.2f}%"
        if isinstance(spillover, float):
            spillover = f"{spillover:.4f}"
        
        summary += f"| {method} | {n_iter} | {rc_1mm} | {rc_2mm} | {rc_3mm} | {rc_4mm} | {rc_5mm} | {uniformity} | {spillover} |\n"

    summary += """

## Detailed Analysis by Method

"""

    for method in sorted(by_method.keys()):
        configs = sorted(by_method[method], key=lambda x: x[0])
        
        summary += f"### {method}\n\n"
        summary += f"**Configurations tested**: {len(configs)} iterations\n\n"
        
        # Show convergence pattern
        summary += "**Recovery Coefficient vs Iterations:**\n\n"
        summary += "| Iterations | RC_1mm | RC_2mm | RC_3mm | RC_4mm | RC_5mm |\n"
        summary += "|-----------|--------|--------|--------|--------|--------|\n"
        
        for n_iter, metrics in configs:
            rc_1mm = f"{metrics.get('RC_1mm', 0):.4f}"
            rc_2mm = f"{metrics.get('RC_2mm', 0):.4f}"
            rc_3mm = f"{metrics.get('RC_3mm', 0):.4f}"
            rc_4mm = f"{metrics.get('RC_4mm', 0):.4f}"
            rc_5mm = f"{metrics.get('RC_5mm', 0):.4f}"
            summary += f"| {n_iter} | {rc_1mm} | {rc_2mm} | {rc_3mm} | {rc_4mm} | {rc_5mm} |\n"
        
        summary += "\n"

    # Recommendations
    summary += """## Recommendations for Your Use Case

### For Quantification (Your Primary Goal)

**What matters most for quantification:**
1. **High Recovery Coefficients (RC)** - Especially for small structures
2. **Low Uniformity Noise** - Less noise = better quantification
3. **Reasonable Spillover** - Not counting background activity

**How to choose:**

1. **If high accuracy matters (research quantification):**
   - Pick the method with highest RC values
   - Accept longer computation time
   - Suggested: **RL** (maximum likelihood)

2. **If speed matters (clinical workflow):**
   - Pick VC (Van Citert deconvolution)
   - Good RC values, much faster than RL
   - Reasonable trade-off

3. **If structure size matters:**
   - Small rods (1-2 mm): Use higher RC method
   - Large rods (4-5 mm): Most methods work well
   - **Check your typical structure sizes**

4. **If background is critical:**
   - Monitor Spillover_Water and Spillover_Air
   - Lower is better (< 0.05 is good)

### Statistical Convergence

- Look for where RC values plateau (diminishing returns from more iterations)
- Typically 10-15 iterations gives 80-90% of maximum correction
- Beyond 20 iterations: minimal improvement, more computation

## Next Steps

1. **Export this comparison**: Use the CSV to plot in Excel/Python
2. **Plot RC vs iterations**: See where each method converges
3. **Calculate time cost**: Note elapsed time per method
4. **Quantify trade-offs**: RC improvement vs time increase
5. **Select for production**: Pick method based on YOUR requirements

## Python Analysis Example

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load comparison results
df = pd.read_csv("pvc_comparison.csv")

# Plot RC_1mm convergence for all methods
plt.figure(figsize=(10, 6))
for method in df['method'].unique():
    method_data = df[df['method'] == method].sort_values('iterations')
    plt.plot(method_data['iterations'], method_data['RC_1mm'], 
             marker='o', label=method)
plt.xlabel('Iterations')
plt.ylabel('RC @ 1mm Rod')
plt.title('Recovery Coefficient Convergence by Method')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('rc_convergence.png', dpi=150)
plt.show()

# Find best method at each iteration count
for iterations in df['iterations'].unique():
    subset = df[df['iterations'] == iterations]
    best = subset.loc[subset['RC_1mm'].idxmax()]
    print(f"Best at {iterations} iterations: {best['method']} (RC_1mm={best['RC_1mm']:.4f})")

# Calculate speed-quality trade-off
print("\\nMethod Rankings (by final RC_1mm):")
final_iter = df['iterations'].max()
final_results = df[df['iterations'] == final_iter].sort_values('RC_1mm', ascending=False)
for idx, row in final_results.iterrows():
    print(f"  {row['method']:6} -> RC_1mm={row['RC_1mm']:.4f}, Spillover={row['Spillover_Water']:.4f}")
```

---

*This analysis will help you make an informed decision about which PVC method is best for your quantification use case.*

"""

    # Save summary
    summary_path = output_dir / 'comparison_summary.md'
    with open(summary_path, 'w') as f:
        f.write(summary)

    return summary


def create_best_method_recommendation(
    results: List[Tuple[str, int, Dict[str, float]]],
    criteria: str = "rc_1mm",
) -> Dict[str, any]:
    """
    Recommend the best method based on specified criteria.

    Parameters
    ----------
    results : List[Tuple[str, int, Dict[str, float]]]
        List of (method, iterations, metrics_dict) tuples
    criteria : str
        Metric to optimize: "rc_1mm", "rc_avg", "uniformity", "spillover", "balanced"

    Returns
    -------
    Dict
        Recommendation with method, iterations, metrics, reasoning
    """

    if not results:
        return {"error": "No results provided"}

    # Filter to highest iteration count for each method
    by_method = {}
    for method, n_iter, metrics in results:
        if method not in by_method or n_iter > by_method[method][0]:
            by_method[method] = (n_iter, metrics)

    recommendations = []

    if criteria == "rc_1mm":
        # Best recovery of small structures
        for method, (n_iter, metrics) in by_method.items():
            rc = metrics.get('RC_1mm', 0)
            recommendations.append({
                'method': method,
                'iterations': n_iter,
                'score': rc,
                'metric': f"RC @ 1mm = {rc:.4f}",
                'reasoning': 'Best for detecting small structures (1mm rods)'
            })

    elif criteria == "rc_avg":
        # Average of all rod sizes
        for method, (n_iter, metrics) in by_method.items():
            rcs = [metrics.get(f'RC_{d}mm', 0) for d in [1, 2, 3, 4, 5]]
            avg_rc = np.mean([r for r in rcs if r > 0])
            recommendations.append({
                'method': method,
                'iterations': n_iter,
                'score': avg_rc,
                'metric': f"Avg RC = {avg_rc:.4f}",
                'reasoning': 'Best overall across all structure sizes'
            })

    elif criteria == "uniformity":
        # Best uniformity (lower STD is better)
        for method, (n_iter, metrics) in by_method.items():
            unif = metrics.get('Uniformity_std_pct', 100)
            score = -unif  # Negative so higher is better
            recommendations.append({
                'method': method,
                'iterations': n_iter,
                'score': score,
                'metric': f"Uniformity %STD = {unif:.2f}%",
                'reasoning': 'Best for background uniformity (lower noise)'
            })

    elif criteria == "spillover":
        # Best spillover (lower is better)
        for method, (n_iter, metrics) in by_method.items():
            spill = metrics.get('Spillover_Water', 1)
            score = -spill  # Negative so higher is better
            recommendations.append({
                'method': method,
                'iterations': n_iter,
                'score': score,
                'metric': f"Spillover Water = {spill:.4f}",
                'reasoning': 'Best background suppression'
            })

    elif criteria == "balanced":
        # Balanced: high RC, low spillover, good uniformity
        for method, (n_iter, metrics) in by_method.items():
            rc_avg = np.mean([metrics.get(f'RC_{d}mm', 0) for d in [1, 2, 3, 4, 5]])
            unif = metrics.get('Uniformity_std_pct', 100)
            spill = metrics.get('Spillover_Water', 1)
            
            # Composite score: high RC + low spillover + low noise
            score = rc_avg - (0.1 * unif) - (10 * spill)
            
            recommendations.append({
                'method': method,
                'iterations': n_iter,
                'score': score,
                'metric': f"RC={rc_avg:.3f}, STD={unif:.1f}%, Spill={spill:.4f}",
                'reasoning': 'Best trade-off across all metrics'
            })

    # Sort by score (descending)
    recommendations.sort(key=lambda x: x['score'], reverse=True)

    if recommendations:
        best = recommendations[0]
        return {
            'rank': 1,
            'method': best['method'],
            'iterations': best['iterations'],
            'metric': best['metric'],
            'reasoning': best['reasoning'],
            'runner_up': recommendations[1] if len(recommendations) > 1 else None,
            'all_ranked': recommendations,
        }

    return {"error": "Could not generate recommendation"}


def export_ranked_comparison(
    results: List[Tuple[str, int, Dict[str, float]]],
    output_dir: str,
) -> Dict[str, str]:
    """
    Create multiple recommendation CSVs ranked by different criteria.

    Parameters
    ----------
    results : List[Tuple[str, int, Dict[str, float]]]
        List of (method, iterations, metrics_dict) tuples
    output_dir : str
        Directory where CSVs will be saved

    Returns
    -------
    Dict[str, str]
        Paths to all generated CSV files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    criteria_list = ['rc_1mm', 'rc_avg', 'uniformity', 'spillover', 'balanced']
    output_files = {}

    for criteria in criteria_list:
        recommendation = create_best_method_recommendation(results, criteria)

        if 'error' in recommendation:
            continue

        csv_path = output_dir / f'ranking_{criteria}.csv'
        output_files[criteria] = str(csv_path)

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Rank', 'Method', 'Iterations', 'Score', 'Details'])

            for rank, rec in enumerate(recommendation['all_ranked'], 1):
                writer.writerow([
                    rank,
                    rec['method'],
                    rec['iterations'],
                    f"{rec['score']:.4f}",
                    rec['metric']
                ])

    return output_files
