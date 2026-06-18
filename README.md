# CCMS Project

## CI/CD Deployment with GitHub Actions and Watchtower

This project is configured to automatically build and publish a Docker image to the GitHub Container Registry (GHCR) whenever code is pushed to the `main` or `production` branches.

### Watchtower Setup (Ubuntu Server)

Watchtower is used on the server to automatically detect when a new image is pushed to GHCR, pull the new image, and gracefully restart the `django_app` container without requiring any SSH deployment steps in GitHub Actions.

**Important**: After GitHub Actions pushes the new image, Watchtower on Ubuntu will handle pulling and restarting.

Here is an example `docker-compose.yml` snippet to run Watchtower on your Ubuntu server:

```yaml
version: '3.8'

services:
  watchtower:
    image: containrrr/watchtower
    container_name: watchtower
    restart: always
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - /root/.docker/config.json:/config.json:ro # For GHCR authentication
    command: --interval 60 --cleanup
```

To authenticate Watchtower with GHCR, make sure you have logged in to GHCR on your Ubuntu server:

```bash
echo $CR_PAT | docker login ghcr.io -u USERNAME --password-stdin
```

This saves the credentials in `~/.docker/config.json` (or `/root/.docker/config.json`), which Watchtower mounts to pull private images from GHCR.

## Running the Application

1. Make sure to create a `.env` file containing your Django environment variables (e.g. `SECRET_KEY`, `DEBUG`, database credentials).
2. Start the application along with Watchtower:

```bash
docker-compose up -d
```
