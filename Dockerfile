# Static-only demo (browser does the inference) — just needs a file server
# with the same cross-origin-isolation headers the current nginx location
# block sets, so onnxruntime-web can use SharedArrayBuffer.
FROM nginx:1.27-alpine

# Served under /pose-tracker/ (not html root): the load balancer forwards
# the full incoming path unchanged to Serverless NEG backends — no
# url_rewrite available there in practice (confirmed for real; see
# bradjobe-dev-infra's lb.tf) — so the container has to physically mirror
# the URL prefix itself, same as ai-hub/ai-tools already do. All of this
# app's own asset references are relative (checked directly), so this
# needed no HTML/JS changes.
COPY frontend/pose-tracker/ /usr/share/nginx/html/pose-tracker/
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 8080
