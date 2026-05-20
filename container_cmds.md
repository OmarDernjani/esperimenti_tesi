# DOCKER
- First time: download the model (after launching ollama container)
    ```sh
    docker compose up ollama -d
    docker exec ollama ollama pull llama3.2
    ```
- Just experiment (ollama container must be up)
    ```sh
    docker compose run --rm -d app python main.py
    docker compose run --rm -d app python resume_missing.py
    ```
- Read logs
    ```sh
    docker logs -f <container_id>
    ```
- Shut down all the containers
    ```sh
    docker compose down
    ```

# PODMAN
See `.sh` scripts launching ollama and experiment containers
- First time: download the model (after launching ollama container)
    ```sh
    ./run_ollama.sh
    podman exec ollama ollama pull qwen2.5:7b
    ```
- Just experiment (ollama container must be up)
    ```sh
    ./run_app.sh
    ./run_resume.sh
    ```
- Read logs
    ```sh
    podman logs -f experiment
    ```
- Shut down all the containers
    ```sh
    podman stop ollama 
    podman stop experiment
    ```