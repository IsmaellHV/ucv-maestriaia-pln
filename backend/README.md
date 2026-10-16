# Backend

API FastAPI de clasificación de noticias en español. El punto de entrada es
`app.py`. Consulta también el [README del proyecto](../README.md).

## Instalación y arranque

Requiere Python 3.12.x y los pesos del modelo descargados con Git LFS.
Desde `backend/`, en PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe app.py
```

Escucha por defecto en **http://127.0.0.1:8000**. El modelo se carga una sola vez
al iniciar. El backend no sirve la interfaz web; inicia el [frontend](../frontend/README.md)
en otra terminal.

| Método | Ruta | Función |
| --- | --- | --- |
| GET | `/health` | Estado del servicio y del modelo |
| GET | `/model-info` | Identidad y configuración del modelo |
| POST | `/predict` | Clasificación de `headline` y `text` |
| GET | `/docs` | Documentación interactiva de la API |
| GET | `/openapi.json` | Contrato OpenAPI |

Ejemplo del cuerpo de `/predict`:

```json
{"headline": "Titular opcional", "text": "Cuerpo obligatorio de la noticia"}
```

## Configuración

`app.py` carga `backend/.env` antes de construir la aplicación. Las variables del
entorno tienen prioridad sobre ese archivo. No subas `.env` al repositorio.

| Variable | Valor del ejemplo | Uso |
| --- | --- | --- |
| `FAKENEWS_MODEL_METADATA` | `results/final_model.json` | Metadata del modelo; ruta relativa a `backend/` |
| `FAKENEWS_DEVICE` | `cpu` | Dispositivo: `cpu`, `auto`, `cuda` o `cuda:0` |
| `FAKENEWS_CORS_ORIGINS` | Vacío | Orígenes permitidos separados por comas; necesarios si el navegador llama directamente a otro origen |

Host y puerto se cambian con argumentos, no con el `.env`:

```powershell
.\.venv\Scripts\python.exe app.py --host 0.0.0.0 --port 8001
```

Si cambias el puerto, actualiza también `target` en `frontend/vite.config.ts`.
Con el proxy local `/api`, CORS puede quedar vacío. Para un frontend publicado
en otro origen, indica su origen exacto, por ejemplo `https://noticias.example.com`,
sin barra final, y reinicia la API. No se admite `*`.

El checkpoint se resuelve desde `results/final_model.json` y se valida su hash.
Los datasets no se leen para inferencia. El sistema clasifica patrones textuales;
no realiza verificación factual.

## Pruebas

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-test.txt
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_inference.py -v
```
