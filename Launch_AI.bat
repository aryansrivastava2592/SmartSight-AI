@echo off
:: Navigate to your project folder (optional if bat is already in the folder)
cd /d "%~dp0"

:: Activate the virtual environment
call venv\Scripts\activate

:: Run the Streamlit app
python -m streamlit run app.py

pause