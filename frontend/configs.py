from dotenv import dotenv_values

venv = dotenv_values(".env")

BACKEND_URL = venv.get("BACKEND_URL")