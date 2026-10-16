# Frontend

Interfaz React + TypeScript + Vite. Consulta también el [README principal](../README.md)
y el [README del backend](../backend/README.md).

## Instalación y arranque

Requiere Node.js compatible con Vite (20.19+ o 22.12+). Desde `frontend/`:

```powershell
Copy-Item .env.example .env
npm ci
npm run dev
```

La API debe estar iniciada en otra terminal mediante `backend/app.py`.

| Servicio | Dirección por defecto |
| --- | --- |
| Interfaz en desarrollo: `npm run dev` | `http://127.0.0.1:5173` |
| Vista previa del build: `npm run preview` | `http://127.0.0.1:4173` |
| Backend al que apunta el proxy | `http://127.0.0.1:8000` |

Vite exige que los puertos configurados estén libres; no elige otro automáticamente.

## Configuración e integración

`VITE_API_BASE_URL=/api` en `.env.example` utiliza el proxy de Vite.
Las llamadas `/api/health`, `/api/model-info` y `/api/predict` se redirigen al
backend retirando el prefijo `/api`. El navegador permanece en el mismo origen.

El destino del proxy y los puertos de desarrollo y vista previa se configuran
en `vite.config.ts`. Si cambias el puerto del backend, actualiza allí `target`.
Reinicia Vite después de modificar su configuración o el `.env`.

## Compilación y publicación

```powershell
npm run build
npm run preview
```

El build se genera en `dist/`. La vista previa permite comprobarlo localmente.
El proxy de Vite no forma parte de los archivos compilados. Para publicar `dist/`,
configura el servidor para redirigir `/api/*` al backend retirando `/api`, o fija
`VITE_API_BASE_URL` con la URL pública de la API antes de ejecutar `npm run build`.
En ese segundo caso, configura `FAKENEWS_CORS_ORIGINS` en el backend con el origen
exacto del frontend y reinicia la API.

Las variables `VITE_*` son públicas y se incorporan al build; no deben contener
secretos. Los cambios de URL en producción requieren volver a compilar.

## Comprobaciones

```powershell
npm test
npm run lint
npm run build
```
