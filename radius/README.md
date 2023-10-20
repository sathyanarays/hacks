## Create k3d cluster

`k3d cluster create`

## Create first-app

`mkdir first-app`
`cd first-app`
`rad init`

## Manually loading the images

```
docker pull ghcr.io/radius-project/deployment-engine:0.26 && \
    docker pull ghcr.io/radius-project/applications-rp:0.26 && \
    docker pull ghcr.io/radius-project/ucpd:0.26 && \
    docker pull ghcr.io/radius-project/controller:0.26

k3d image import ghcr.io/radius-project/deployment-engine:0.26 && \
    k3d image import ghcr.io/radius-project/applications-rp:0.26 && \
    k3d image import ghcr.io/radius-project/ucpd:0.26 && \
    k3d image import ghcr.io/radius-project/controller:0.26
```
