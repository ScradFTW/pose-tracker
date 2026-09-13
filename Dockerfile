# Static-only demo (browser does the inference) — just needs a file server
# with the same cross-origin-isolation headers the current nginx location
# block sets, so onnxruntime-web can use SharedArrayBuffer.
FROM nginx:1.27-alpine

COPY frontend/pose-tracker/ /usr/share/nginx/html/
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 8080
