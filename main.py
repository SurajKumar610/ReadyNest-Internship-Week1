import os
import sys
import argparse
import subprocess
import pandas as pd

# Import local modules
from src.data_loading import load_from_file, get_dataset_metadata
from src.data_cleaning import remove_duplicates, impute_missing_values, handle_outliers
from src.descriptive_statistics import calculate_descriptive_statistics
from src.insight_extraction import extract_insights
from src.forecasting import time_series_forecast
from src.reporting import generate_pdf_report, generate_excel_report, generate_ppt_report

def run_pipeline(input_file, output_dir="reports", forecast_col=None, date_col=None):
    """
    Orchestrate the entire analytical pipeline from loading to generating reports.
    """
    print(f"\n[Info] Starting analytical pipeline for: {input_file}")
    
    # 1. Loading
    file_ext = input_file.split('.')[-1]
    df_raw = load_from_file(input_file, file_ext)
    print(f"[Success] Loaded raw dataset of shape: {df_raw.shape[0]} rows x {df_raw.shape[1]} columns.")
    
    # 2. Basic Cleaning (Duplicate removal, automatic numeric imputation)
    df_cleaned = remove_duplicates(df_raw)
    numeric_cols = df_cleaned.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        df_cleaned = impute_missing_values(df_cleaned, numeric_cols, strategy='median')
        print(f"[Success] Removed duplicates and imputed missing numeric values using column medians.")
        
    # 3. Descriptive Stats
    print("[Stats] Generating descriptive statistics summary...")
    stats_df = calculate_descriptive_statistics(df_cleaned)
    
    # 4. Insight Extraction
    print("[Insights] Analyzing patterns and extracting key findings...")
    insights = extract_insights(df_cleaned)
    print(f"[Success] Successfully extracted {len(insights['findings'])} findings.")
    
    # 5. Optional forecasting
    if forecast_col and date_col:
        print(f"[Forecast] Fitting ARIMA time series model on {forecast_col}...")
        try:
            fore_res = time_series_forecast(df_cleaned, date_col, forecast_col, forecast_steps=6)
            print(f"[Success] Generated 6-step forecast.")
            # Append forecast summary to findings
            insights["findings"].append(f"Forecasting: Model projects future values. Details exported to output worksheets.")
        except Exception as e:
            print(f"[Warning] Forecasting skipped: {str(e)}")
            
    # 6. Report Generation
    os.makedirs(output_dir, exist_ok=True)
    pdf_out = os.path.join(output_dir, "pdf", "executive_report.pdf")
    excel_out = os.path.join(output_dir, "excel", "dataset_sheets.xlsx")
    ppt_out = os.path.join(output_dir, "ppt", "executive_slides.pptx")
    
    print("[Report] Generating report documents (PDF, Excel, PPT)...")
    generate_pdf_report(stats_df, insights, pdf_out)
    generate_excel_report(df_raw, df_cleaned, stats_df, excel_out)
    generate_ppt_report(insights, ppt_out)
    
    print("\n[Done] Pipeline Execution Completed Successfully!")
    print(f"Reports exported to: {os.path.abspath(output_dir)}/")
    print(f"  - PDF Report: {pdf_out}")
    print(f"  - Excel Workbook: {excel_out}")
    print(f"  - PowerPoint Deck: {ppt_out}")

def main():
    parser = argparse.ArgumentParser(description="Dataset Analysis & Reporting Platform Orchestrator")
    parser.add_argument("-d", "--dashboard", action="store_true", help="Launch the interactive Streamlit dashboard UI")
    parser.add_argument("-i", "--input", type=str, help="Path to input dataset file (CSV/Excel/JSON) for command-line execution")
    parser.add_argument("-o", "--output-dir", type=str, default="reports", help="Directory where generated reports should be stored")
    parser.add_argument("-f", "--forecast-col", type=str, help="Value column to perform time series forecasting on")
    parser.add_argument("-tc", "--time-col", type=str, help="Date/Time column representing timestamps for forecasting")
    parser.add_argument("-s", "--schedule", type=float, help="Schedule period in minutes to run report generation repeatedly. E.g. --schedule 60 for hourly, or 1440 for daily.")
    
    args = parser.parse_args()
    
    if args.dashboard:
        # Launch Streamlit dashboard in a subprocess
        dashboard_path = os.path.join("dashboard", "streamlit_app.py")
        if not os.path.exists(dashboard_path):
            print(f"Error: Dashboard script not found at {dashboard_path}")
            sys.exit(1)
            
        print("Starting Streamlit dashboard UI...")
        try:
            # Run using the virtualenv python/streamlit if present
            venv_bin_path = os.path.join(".venv", "Scripts", "streamlit") if sys.platform.startswith("win") else os.path.join(".venv", "bin", "streamlit")
            if os.path.exists(venv_bin_path + ".exe") or os.path.exists(venv_bin_path):
                cmd = [venv_bin_path, "run", dashboard_path]
            else:
                cmd = ["streamlit", "run", dashboard_path]
                
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\nDashboard stopped.")
        except Exception as e:
            print(f"Failed to launch dashboard: {str(e)}")
            
    elif args.input:
        if not os.path.exists(args.input):
            print(f"Error: Input file '{args.input}' not found.")
            sys.exit(1)
            
        if args.schedule:
            import time
            print(f"[Schedule] Starting scheduled report generation every {args.schedule} minutes...")
            try:
                while True:
                    run_pipeline(args.input, args.output_dir, args.forecast_col, args.time_col)
                    print(f"[Schedule] Run completed. Next run scheduled in {args.schedule} minutes. Press Ctrl+C to exit.")
                    time.sleep(args.schedule * 60)
            except KeyboardInterrupt:
                print("\n[Schedule] Report scheduler stopped.")
        else:
            run_pipeline(args.input, args.output_dir, args.forecast_col, args.time_col)
    else:
        # No arguments provided, show help or suggest running with dashboard
        parser.print_help()
        print("\nTip: Run 'python main.py --dashboard' to start the interactive web application.")

if __name__ == "__main__":
    main()
