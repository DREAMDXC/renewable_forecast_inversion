Renewable Energy Forecast Inversion
📌 Project Description
This project provides a dynamic weighted smoothing approach to generate renewable energy forecast data with specified prediction errors. It enables systematic analysis of forecast accuracy impact on power systems, addressing the lack of comprehensive forecast datasets in practical research.

🎯 Features
Wind & Solar Forecast Inversion - Separate algorithms for wind and photovoltaic power

Adaptive Error Control - Automatically adjusts smoothing parameters to match target MAE

Dual Smoothing Modes:

Temporal - For wind power (adjacent time points)

Daily - For solar power (same hours across days)

Visual Analysis - Kernel width analysis and forecast comparison plots

📊 Outputs
*_kernel_analysis.xlsx - Kernel width vs MAE analysis

traverse_results.png - Kernel width analysis plot

adaptive_smoothing_comparison.png - Original vs smoothed data

🛠 Requirements
python, pandas, numpy, matplotlib, scikit-learn

📄 Data Format
Input: Excel file with power data in second column (automatically normalized by capacity).

🎯 Applications
Power system reliability assessment

Forecast error impact analysis

Renewable energy integration studies

Power market simulation
