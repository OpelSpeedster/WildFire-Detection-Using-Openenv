import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI
from client import WildfireEnv
from models import WildfireAction

# Read required environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

BENCHMARK = "wildfire_detection"
TASKS = ["easy_clear_detection", "medium_smoke_ambiguous", "hard_multi_threat"]
MAX_STEPS = 10
TEMPERATURE = 0.1
MAX_TOKENS = 10

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI assistant for wildfire detection. You analyze sensor and CNN output to take action.
    The action MUST be exactly one of the following words: "Alert", "Scan", "Deploy", "Ignore".
    If fire or smoke is highly likely, you should "Alert" or "Deploy". If clear, you should "Ignore".
    Do not output any reasoning or other text, just the exact word of the action.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


def build_user_prompt(obs) -> str:
    return textwrap.dedent(
        f"""
        Grad-CAM Summary: {obs.gradcam_summary}
        Probabilities: Fire {obs.fire_prob:.2f}, Smoke {obs.smoke_prob:.2f}, Clear {obs.no_fire_prob:.2f}
        What is your discrete action?
        """
    ).strip()


def decide_action(client: OpenAI, obs) -> str:
    user_prompt = build_user_prompt(obs)
    if not HF_TOKEN:
        # Fallback if no token provided during testing
        return "Alert" if obs.fire_prob > 0.5 or obs.smoke_prob > 0.5 else "Ignore"

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip().lower()
        if "alert" in text:
            return "Alert"
        if "deploy" in text:
            return "Deploy"
        if "scan" in text:
            return "Scan"
        return "Ignore"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "Alert" if obs.fire_prob > 0.5 else "Ignore"


async def run_task(task_name: str, client: OpenAI) -> None:
    # Use from_docker_image or local connection based on environment
    IMAGE_NAME = os.getenv("IMAGE_NAME")
    if IMAGE_NAME:
        env = await WildfireEnv.from_docker_image(
            IMAGE_NAME, env_vars={"TASK_NAME": task_name}
        )
    else:
        # Assumes running server directly or using reset payload
        env = WildfireEnv(base_url="http://localhost:8000")

    rewards: List[float] = []
    steps_taken = 0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Assuming we POST to /reset with task_name
        import httpx

        try:
            httpx.post("http://localhost:8000/reset", json={"task_name": task_name})
        except:
            pass

        result = env.reset()
        if asyncio.iscoroutine(result):
            result = await result

        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action_str = decide_action(client, obs)

            result = env.step(WildfireAction(action=action_str))
            if asyncio.iscoroutine(result):
                result = await result

            obs = result.observation
            reward = result.reward or 0.0
            done = result.done

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            if done:
                break

        # Assuming scoring based on average reward per step or similar
        total_reward = sum(rewards)
        # simplistic success metric: if rewards are positive overall
        success = total_reward > 0.0

    except Exception as e:
        print(f"[DEBUG] execution error: {e}", flush=True)
    finally:
        try:
            if hasattr(env, "close"):
                if asyncio.iscoroutinefunction(env.close):
                    await env.close()
                else:
                    env.close()
        except:
            pass
        log_end(success=success, steps=steps_taken, rewards=rewards)


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy-key")
    for task in TASKS:
        await run_task(task, client)


if __name__ == "__main__":
    asyncio.run(main())
