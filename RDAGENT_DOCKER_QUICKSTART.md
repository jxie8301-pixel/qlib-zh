# RD-Agent Docker Quick Start

This workspace now includes a Docker-based environment for Qlib + RD-Agent.

## 1. Prepare the environment file

Copy the example file:

- `.env.example` → `.env`

Then fill in your model provider credentials.

The default setup in this workspace is now configured for Zhipu GLM:

- `CHAT_MODEL=glm-4.7-flash`
- `OPENAI_API_BASE=https://open.bigmodel.cn/api/paas/v4`
- `EMBEDDING_MODEL=Embedding-3`

## 2. Build the image

The image is already configured to build from the local workspace.

- Stable Qlib install: set `IS_STABLE=yes`
- Local source install: set `IS_STABLE=no`

The current compose file uses `IS_STABLE=no` so the local source is used.

## 3. Start a container shell

Use Docker Compose with the bundled service:

- `docker compose run --rm rdagent bash`

## 4. Run factor mining

Inside the container, run:

- `rdagent fin_factor`

## 5. Useful commands

- Check the CLI help: `rdagent --help`
- Fin-quant workflow: `rdagent fin_quant`
- Factor-from-reports workflow: `rdagent fin_factor_report`

## 6. Notes

- The container mounts the current repository at `/qlib`.
- The folder `git_ignore_folder/` is mounted for logs and temporary data.
- You still need valid model credentials before the agent can run.
