# Static-only demo (browser does the inference) — just needs a file server
# with the same cross-origin-isolation headers the current nginx location
# block sets, so onnxruntime-web can use SharedArrayBuffer.
#
# vendor/ is deliberately NOT copied here: those files (onnxruntime-web's
# WASM runtime + the 61MB trained model) go to a GCS backend bucket
# instead (bradjobe-dev-infra's pose_tracker_assets.tf) — Cloud Run
# enforces a 32MB response size limit, confirmed for real against
# pose_net.onnx (nginx served it fully; the client still saw a 500 from
# Cloud Run's own proxy cutting the response short). cloudbuild.yaml
# uploads vendor/ straight to that bucket instead of into this image.
FROM nginx:1.27-alpine

# Served under /pose-tracker/ (not html root): the load balancer forwards
# the full incoming path unchanged to Serverless NEG backends — no
# url_rewrite available there in practice (confirmed for real; see
# bradjobe-dev-infra's lb.tf) — so the container has to physically mirror
# the URL prefix itself, same as ai-hub/ai-tools already do. All of this
# app's own asset references are relative (checked directly), so this
# needed no HTML/JS changes.
COPY frontend/pose-tracker/index.html frontend/pose-tracker/app.js frontend/pose-tracker/style.css /usr/share/nginx/html/pose-tracker/
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 8080
