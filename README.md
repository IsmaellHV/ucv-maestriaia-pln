# Clasificador de noticias en español

Aplicación académica con backend FastAPI y frontend React + TypeScript + Vite.
Clasifica patrones textuales como `True = 0` o `Fake = 1`; **no verifica hechos**.
Las puntuaciones softmax no son probabilidades objetivas de veracidad.

## Estructura y exposición

```text
backend/                  API Python y lógica compartida de inferencia
  .env.example            Configuración sin credenciales
  app.py                  Punto de entrada con carga de .env
  src/fakenews_es/         Código original de inferencia y módulos que importa
  results/final_model.json Identidad y hash del modelo congelado
  results/runs/.../checkpoint-best/ Pesos y tokenizer del modelo final
  tests/                  Pruebas de inferencia y contrato HTTP
frontend/                 Aplicación web independiente y su .env.example
dataset/                  Excel originales y atribución del corpus
MANIFEST.json             SHA-256 de los archivos originales copiados
```

Son dos procesos separados. FastAPI no publica HTML ni los archivos del frontend.

Instrucciones por componente: [backend](backend/README.md) y [frontend](frontend/README.md).

| Servicio | Dirección local | Rutas |
| --- | --- | --- |
| Backend | `http://127.0.0.1:8000` | `GET /health`, `GET /model-info`, `POST /predict`, `/docs` |
| Frontend | `http://127.0.0.1:5173` | Interfaz web; proxy `/api/*` hacia el backend |
| Vista previa del frontend | `http://127.0.0.1:4173` | Build local mediante `npm run preview` |

Host y puerto del backend: `python app.py --host 127.0.0.1 --port 8000`, desde
`backend/` y con el entorno activado. Si cambias ese puerto, actualiza `target`
en `frontend/vite.config.ts`. Ese archivo también fija los puertos del frontend.
Vite exige que sus puertos estén libres; no cambia automáticamente a otro.

## Iniciar el backend

Requiere Python 3.12.x. Desde esta carpeta, en PowerShell:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe app.py
```

El archivo `.env` se carga antes de crear FastAPI. Las variables ya definidas en el
entorno tienen prioridad. El ejemplo usa CPU; `auto` permite GPU si la instalación
de PyTorch y el equipo son compatibles.

La API carga el modelo una vez al arrancar, verifica el SHA-256 y utiliza el
tokenizer local y los mismos módulos de preparación de entrada del experimento.
No necesita leer los datasets ni descargar pesos para atender solicitudes.
Los módulos compartidos `beto.py`, `data.py`, `evaluation.py` y `experiment.py`
se conservan para reutilizar las reglas originales; el arranque no entrena.

## Iniciar el frontend

Requiere Node.js compatible con el Vite del proyecto (20.19+ o 22.12+).
En otra terminal, desde la carpeta principal:

```powershell
cd frontend
Copy-Item .env.example .env
npm ci
npm run dev
```

Abre `http://127.0.0.1:5173`. `VITE_API_BASE_URL=/api` utiliza el proxy local de
`vite.config.ts` hacia el puerto 8000.

Para compilar: `npm run build`. El resultado es `frontend/dist/`.
El proxy de Vite sólo existe en desarrollo y `npm run preview`; al publicar el
frontend estático se necesita una redirección `/api/*` al backend retirando `/api`,
o definir `VITE_API_BASE_URL` con la URL pública del backend **antes de compilar**.
En el segundo caso, configurar `FAKENEWS_CORS_ORIGINS` con el origen exacto del
frontend y reiniciar la API. Las variables `VITE_*` son visibles en el navegador.
`--host 0.0.0.0` permite al backend escuchar fuera del equipo cuando se despliegue.

## Datasets

`train.xlsx`, `development.xlsx` y `test.xlsx` se copian sin modificaciones.
Se conserva [el README original del corpus](dataset/README.md), con autores,
referencias y licencia CC BY 4.0 indicada por sus autores.
Se incluyen como material de la entrega; no son un requisito de inferencia.
No usar test para seleccionar o volver a ajustar el modelo.
La clasificación usa únicamente `Headline` y `Text`.

## Pasar a un repositorio nuevo

Copia **el contenido** de esta carpeta, incluidos los archivos que empiezan con
punto, a una carpeta vacía. No copies el `.git` del proyecto académico.

El archivo `model.safetensors` ocupa aproximadamente 419 MiB y se configura para
Git LFS mediante `.gitattributes`. Instala Git LFS antes del primer commit:

```powershell
git init -b production
git lfs install
git add .
git lfs ls-files
git commit -m "Publicar aplicación de clasificación de noticias"
```

Comprueba que `git lfs ls-files` incluya `model.safetensors`. Después de crear el
repositorio público vacío en GitHub, agrega su URL como remoto y sube `production`.
Quien clone debe tener Git LFS y ejecutar `git lfs pull` para obtener los pesos.
Los pesos consumen almacenamiento y transferencia LFS de la cuenta.

Esta carpeta no contiene historial Git, documentos académicos, entornos virtuales,
`node_modules`, archivos `.env` reales ni otras ejecuciones experimentales.
Los campos de trazabilidad del modelo conservan referencias a artefactos de
evaluación del repositorio académico; esos artefactos no se necesitan para ejecutar
la aplicación y no se incluyen aquí. La licencia del corpus no se extiende
automáticamente al código o a los pesos del modelo.

## Comprobaciones

Desde `backend/`, con las dependencias de pruebas instaladas:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-test.txt
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_inference.py -v
```

Desde `frontend/`: `npm test`, `npm run lint` y `npm run build`.
Los hashes de `MANIFEST.json` permiten verificar las copias, incluidos los datasets
y el checkpoint, sin procesar el corpus.
