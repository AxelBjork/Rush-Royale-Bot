:: Only has to be run once if not run before
python -m venv .bot_env
%LOCALAPPDATA%\Programs\Python\Python39\python -m venv .bot_env
:: if this does not work add the path to where your python installation is located.
call .bot_env\Scripts\activate.bat
pip install -r requirements.txt