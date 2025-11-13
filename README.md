# Game Builder API

This project exposes a FastAPI service that orchestrates Phaser arcade game generation workflows. It is designed so automations (for example Zapier) can drive each step of a content pipeline using OpenAI for ideation and GitHub for code updates.

## Environment variables

Set the following variables before running the service:

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | API key used when calling the OpenAI chat completion endpoint. |
| `OPENAI_MODEL` | *(Optional)* Override the model name. Defaults to `gpt-4o-mini`. |
| `GITHUB_TOKEN` | Token with read/write access to the repositories you wish to update. |
| `TEMPLATE_REPO` | *(Optional)* Repo slug describing the Phaser template that the service references when creating configurations. Defaults to `notr3kt/arcade-template-phaser`. |
| `GIT_AUTHOR_NAME` | *(Optional)* Commit author name to use for automated commits. Defaults to `Game Builder Bot`. |
| `GIT_AUTHOR_EMAIL` | *(Optional)* Commit author email to use for automated commits. Defaults to `bot@example.com`. |

## Installing dependencies

```bash
pip install -r requirements.txt
```

## Running the API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Endpoints

Zapier (or any HTTP client) can call the following endpoints sequentially as part of the game publishing workflow.

### `POST /new-game`

**Request body**

```json
{
  "title": "string",
  "idea": "string"
}
```

The service prompts OpenAI to transform the game idea into a `game_config` object compatible with `public/game_config.json` from the Phaser template. The response includes both the generated configuration and a list of sprite prompts to feed into an image generation step.

**Response body**

```json
{
  "game_config": { "...": "..." },
  "sprite_prompts": ["..."]
}
```

### `POST /update-code`

**Request body**

```json
{
  "repo_full_name": "owner/repo",
  "game_config": { "...": "..." }
}
```

The service clones the given GitHub repository using the provided token, writes `public/game_config.json`, optionally updates `src/main.js` by asking OpenAI to integrate the configuration, then commits and pushes the changes back to the repository.

**Response body**

```json
{ "status": "updated" }
```

### `POST /finalize`

**Request body**

```json
{
  "repo_full_name": "owner/repo",
  "sprite_urls": {
    "key": "https://..."
  }
}
```

Once sprite assets are generated, call this endpoint to merge their URLs into `public/game_config.json.spritePaths`, commit, and push the update.

**Response body**

```json
{ "status": "finalized" }
```

### `GET /health`

Simple health probe that returns `{ "status": "ok" }`.

## Zapier usage outline

1. **Trigger**: Receive a new game idea with title and description.
2. **Action**: Call `POST /new-game` to generate the base configuration and sprite prompts.
3. **Action**: Use the prompts to generate sprite artwork (for example via an image generation Zap). Capture the resulting URLs.
4. **Action**: Call `POST /update-code` to initialize the project repository with the generated configuration and any template adjustments.
5. **Action**: Upload the generated sprites to your hosting provider and gather their URLs.
6. **Action**: Call `POST /finalize` to inject the hosted sprite URLs into the game configuration and publish the final update to GitHub.

Repeat the workflow for each new game pitch.
