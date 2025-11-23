# Deployment Guide (Docker → Azure Container Registry → Container Apps)

## Prerequisites

* Docker installed locally
* Azure CLI installed and logged in
* ACR name: `dressupcontainerregistry`
* Image name: `dressup-ai-tagging`
* Tag: `latest`

---

## 0. Move to folder

```bash
cd app
```

## 1. Build Docker Image

```bash
docker build -t dressup-ai-tagging:latest .
```

## 1,5. Run on docker

```bash
docker run -d -p 8000:8000 --name dressup-ai-tagging dressup-ai-tagging:latest

docker logs -f dressup-ai-tagging
```

## 2. Tag Image for Azure Container Registry

```bash
docker tag dressup-ai-tagging:latest dressupcontainerregistry.azurecr.io/dressup-ai-tagging:latest
```

## 3. Login to Azure Container Registry

```bash
az acr login --name dressupcontainerregistry
```

## 4. Push Image to ACR

```bash
docker push dressupcontainerregistry.azurecr.io/dressup-ai-tagging:latest
```

## 5. Redeploy to Azure Container Apps

```bash
az containerapp update --name dressup-ai-tagging --resource-group dressup-backend --image dressupcontainerregistry.azurecr.io/dressup-ai-tagging:latest
```

---

The container app will restart automatically and use the updated image.
