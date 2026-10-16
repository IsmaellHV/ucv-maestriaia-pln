# Validación de la entrega — 16 de octubre de 2026

- 53 archivos copiados conservan sus SHA-256, incluidos los tres datasets,
  el código compartido, la metadata, el tokenizer y los pesos del modelo.
- Backend: seis pruebas de `test_inference.py` aprobadas usando el código
  de esta entrega y el entorno Python existente del proyecto.
- Integración con el checkpoint real en CPU: `/health`, `/model-info` y
  `/predict` respondieron correctamente mediante TestClient; carga única
  del modelo confirmada. Se usó texto sintético, sin procesar los datasets.
- La ruta `/` devuelve 404: FastAPI no sirve el frontend.
- Frontend: instalación con `npm ci`, 11 pruebas aprobadas, lint y build correctos.
- Git reconoce `model.safetensors` como LFS, ignora los `.env` reales y conserva
  ambos `.env.example`.

La comprobación del backend reutilizó las dependencias ya instaladas, con
`python-dotenv` instalado en una carpeta temporal; no se verificó una instalación
Python nueva desde cero. La importación inicial de Transformers fue lenta al
recorrer sus archivos locales. No se publicaron repositorios ni servicios.
Las dependencias y los archivos generados durante las comprobaciones se dejaron
fuera de esta entrega.
