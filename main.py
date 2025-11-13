import asyncio
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from git import Repo
from git.exc import GitError
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
TEMPLATE_REPO = os.getenv("TEMPLATE_REPO", "notr3kt/arcade-template-phaser")

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

app = FastAPI(title="Game Builder API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class NewGameRequest(BaseModel):
    title: str
    idea: str


class NewGameResponse(BaseModel):
    game_config: Dict[str, Any]
    sprite_prompts: List[str] = Field(default_factory=list)


class UpdateCodeRequest(BaseModel):
    repo_full_name: str
    game_config: Dict[str, Any]


class FinalizeRequest(BaseModel):
    repo_full_name: str
    sprite_urls: Dict[str, str]


async def call_openai(messages: List[Dict[str, str]], *, response_format: Optional[str] = None) -> str:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured")

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload: Dict[str, Any] = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": 0.2,
    }
    if response_format:
        payload["response_format"] = {"type": response_format}

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(OPENAI_API_URL, headers=headers, json=payload)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:  # pragma: no cover - network errors
            logger.exception("OpenAI request failed: %s", exc.response.text)
            raise HTTPException(status_code=502, detail="OpenAI request failed") from exc

    data = response.json()
    content = data["choices"][0]["message"]["content"]
    if not content:
        raise HTTPException(status_code=502, detail="OpenAI returned empty response")
    return content


async def call_openai_for_json(prompt: str) -> Dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant that produces strictly valid JSON responses. "
                "Do not include code fences or commentary."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    content = await call_openai(messages, response_format="json_object")
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        logger.exception("Failed to parse OpenAI JSON: %s", content)
        raise HTTPException(status_code=502, detail="OpenAI returned invalid JSON") from exc


async def generate_game_config(title: str, idea: str) -> NewGameResponse:
    template_hint = (
        "Generate game configuration compatible with public/game_config.json in the "
        f"Phaser template repo {TEMPLATE_REPO}."
    )
    prompt = json.dumps(
        {
            "title": title,
            "idea": idea,
            "instructions": (
                template_hint
                + " Return JSON with keys 'game_config' (object) and 'sprite_prompts' (array of strings)."
            ),
        }
    )

    result = await call_openai_for_json(prompt)
    if "game_config" not in result:
        raise HTTPException(status_code=502, detail="OpenAI response missing game_config")

    sprite_prompts = result.get("sprite_prompts") or []
    return NewGameResponse(game_config=result["game_config"], sprite_prompts=sprite_prompts)


def build_repo_url(repo_full_name: str) -> str:
    if not GITHUB_TOKEN:
        raise HTTPException(status_code=500, detail="GITHUB_TOKEN is not configured")
    return f"https://{GITHUB_TOKEN}:x-oauth-basic@github.com/{repo_full_name}.git"


def write_game_config(repo_path: Path, game_config: Dict[str, Any]) -> Path:
    public_dir = repo_path / "public"
    public_dir.mkdir(parents=True, exist_ok=True)
    config_path = public_dir / "game_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(game_config, f, indent=2, ensure_ascii=False)
    return config_path


async def maybe_update_main_js(repo_path: Path, game_config: Dict[str, Any]) -> Optional[Path]:
    main_js = repo_path / "src" / "main.js"
    if not main_js.exists() or not OPENAI_API_KEY:
        return None

    original_code = main_js.read_text(encoding="utf-8")
    prompt = (
        "You are updating a Phaser game's src/main.js file to use a provided configuration. "
        "Integrate the given JSON config as needed (imports, scene configuration, assets). "
        "Return ONLY the full updated JavaScript source code without fences."
    )
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": json.dumps({"game_config": game_config, "current_main_js": original_code}),
        },
    ]
    try:
        content = await call_openai(messages)
    except HTTPException:
        return None

    if content and content.strip():
        main_js.write_text(content.strip(), encoding="utf-8")
        return main_js
    return None


def ensure_git_identity(repo: Repo) -> None:
    name = os.getenv("GIT_AUTHOR_NAME", "Game Builder Bot")
    email = os.getenv("GIT_AUTHOR_EMAIL", "bot@example.com")
    with repo.config_writer() as cw:
        cw.set_value("user", "name", name)
        cw.set_value("user", "email", email)


def commit_and_push(repo: Repo, message: str, paths: Optional[List[Path]] = None) -> None:
    ensure_git_identity(repo)
    if paths:
        repo.index.add([str(path.relative_to(repo.working_tree_dir)) for path in paths])
    else:
        repo.git.add(A=True)

    if not repo.is_dirty(untracked_files=True):
        logger.info("No changes detected after update; skipping commit")
        return

    repo.index.commit(message)
    try:
        origin = repo.remote(name="origin")
        origin.push()
    except GitError as exc:
        logger.exception("Failed to push changes", exc_info=exc)
        raise HTTPException(status_code=502, detail="Failed to push changes to GitHub") from exc


def clone_repo(repo_full_name: str) -> Path:
    repo_url = build_repo_url(repo_full_name)
    base_dir = Path(tempfile.mkdtemp(prefix="game-builder-"))
    repo_dir = base_dir / "repo"
    try:
        Repo.clone_from(repo_url, repo_dir)
    except GitError as exc:
        shutil.rmtree(base_dir, ignore_errors=True)
        logger.exception("Failed to clone repository", exc_info=exc)
        raise HTTPException(status_code=502, detail="Failed to clone repository") from exc
    return repo_dir


def load_game_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def update_sprite_paths(existing_config: Dict[str, Any], sprite_urls: Dict[str, str]) -> Dict[str, Any]:
    updated_config = dict(existing_config)
    sprite_paths = dict(updated_config.get("spritePaths") or {})
    sprite_paths.update(sprite_urls)
    updated_config["spritePaths"] = sprite_paths
    return updated_config


@app.post("/new-game", response_model=NewGameResponse)
async def new_game(payload: NewGameRequest) -> NewGameResponse:
    return await generate_game_config(payload.title, payload.idea)


@app.post("/update-code")
async def update_code(payload: UpdateCodeRequest) -> Dict[str, str]:
    repo_path = await asyncio.to_thread(clone_repo, payload.repo_full_name)
    repo = Repo(repo_path)
    staged_paths: List[Path] = []

    try:
        config_path = write_game_config(repo_path, payload.game_config)
        staged_paths.append(config_path)

        updated_main = await maybe_update_main_js(repo_path, payload.game_config)
        if updated_main:
            staged_paths.append(updated_main)

        await asyncio.to_thread(commit_and_push, repo, "chore: update game configuration", staged_paths)
    finally:
        shutil.rmtree(repo_path.parent, ignore_errors=True)

    return {"status": "updated"}


@app.post("/finalize")
async def finalize(payload: FinalizeRequest) -> Dict[str, str]:
    repo_path = await asyncio.to_thread(clone_repo, payload.repo_full_name)
    repo = Repo(repo_path)

    try:
        config_path = repo_path / "public" / "game_config.json"
        if not config_path.exists():
            raise HTTPException(status_code=400, detail="game_config.json not found in repository")

        existing_config = load_game_config(config_path)
        updated_config = update_sprite_paths(existing_config, payload.sprite_urls)
        write_game_config(repo_path, updated_config)

        await asyncio.to_thread(
            commit_and_push,
            repo,
            "chore: finalize sprite assets",
            [config_path],
        )
    finally:
        shutil.rmtree(repo_path.parent, ignore_errors=True)

    return {"status": "finalized"}


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
