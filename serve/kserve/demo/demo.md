### Generate the model

```
python3 train.py
```

Model gets generated at `mnist_cnn.pt`

### Install Kserve

```
curl -s "https://raw.githubusercontent.com/kserve/kserve/release-0.12/hack/quick_install.sh" | bash
```

### Create Archive

```
torch-model-archiver --model-name mnist --version 1.0 --model-file model.py --serialized-file mnist_cnn.pt --handler mnist_handler.py
```

### Run inference

```
INGRESS_GATEWAY_SERVICE=$(kubectl get svc --namespace istio-system --selector="app=istio-ingressgateway" --output jsonpath='{.items[0].metadata.name}')
kubectl port-forward --namespace istio-system svc/${INGRESS_GATEWAY_SERVICE} 8080:80
```

```
export INGRESS_HOST=localhost
export INGRESS_PORT=8080
```

```
MODEL_NAME=mnist
SERVICE_HOSTNAME=$(kubectl get inferenceservice torchserve -o jsonpath='{.status.url}' | cut -d "/" -f 3)
```

```
curl -v -H "Host: ${SERVICE_HOSTNAME}" -H "Content-Type: application/json" http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/${MODEL_NAME}:predict -d @./input.json
```

### Auto scaling

```
MODEL_NAME=autoscale-test
INPUT_PATH=input.json
SERVICE_HOSTNAME=$(kubectl get inferenceservice $MODEL_NAME -o jsonpath='{.status.url}' | cut -d "/" -f 3)

hey -z 30s -c 5 -m POST -host ${SERVICE_HOSTNAME} -D $INPUT_PATH http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/$MODEL_NAME:predict
```

### Canary

```
kubectl get virtualservices my-model-predictor-ingress -o yaml
```

