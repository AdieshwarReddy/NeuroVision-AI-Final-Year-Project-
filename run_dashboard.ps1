<#
run_dashboard.ps1 — Quick launcher for the NeuroVision AI Streamlit dashboard.
Run from project root: .\run_dashboard.ps1
#>
# PowerShell launcher script for NeuroVision AI Dashboard
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  NeuroVision AI - Brain Tumor MRI Classifier    " -ForegroundColor Cyan
Write-Host "  Streamlit Dashboard Launcher                   " -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting dashboard on http://localhost:8501 ..." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop." -ForegroundColor Yellow
Write-Host ""

# Activate venv and launch streamlit
& ".\.venv\Scripts\python.exe" -m streamlit run app\dashboard.py --server.port 8501 --server.headless false --browser.gatherUsageStats false
