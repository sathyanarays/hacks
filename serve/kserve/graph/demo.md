SERVICE_HOSTNAME=model-chainer.default.example.co

curl -v -H "Host: ${SERVICE_HOSTNAME}" -H "Content-Type: application/json" http://${INGRESS_HOST}:${INGRESS_PORT} -d @./iris-input.json



