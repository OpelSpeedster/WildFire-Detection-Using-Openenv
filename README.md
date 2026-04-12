---
title: Wildfire Detection Environment
emoji: 🔥
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - wildfire-detection
  - reinforcement-learning
  - firenet-cnn
---

# Wildfire Detection RL Environment

An OpenEnv-compatible reinforcement learning environment for autonomous wildfire patrol using the FirenetCNN model. The agent receives sequential camera observations, runs inference, and decides actions to detect fire or smoke as early as possible.

## Features

- **Exact FirenetCNN Inference** - Uses the original model from the Forest-Fire-Detection-Using-FirenetCNN-and-XAI-Techniques repository
- **Grad-CAM XAI** - Explainable AI with heatmap visualization for model decisions
- **Sequential Decision Making** - Agent processes multiple frames and learns from context
- **Reward-based Learning** - Optimizes for early detection while minimizing false alarms

## Quick Start

### Using the Docker Image

```bash
# Pull and run the pre-built image
docker run -p 8000:8000 wildfire-detection
```

### Building from Source

```bash
# Build the Docker image
docker build -t wildfire-detection .

# Run the container
docker run -p 8000:8000 wildfire-detection
```

### Local Development

```bash
# Install dependencies
pip install -e .

# Run the server
python run_server.py
```

## API Usage

### Reset Environment

```bash
curl -X POST http://localhost:8000/reset
```

### Take Action

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": "Alert"}'
```

### Get Schema

```bash
curl http://localhost:8000/schema
```

Access the interactive API docs at http://localhost:8000/docs

## Inference Script

The `inference.py` script runs an LLM agent that makes decisions:

```bash
# Set API key (injected by judges at evaluation)
export API_KEY=your_huggingface_token

# Run inference
python inference.py
```

### Environment Variables

| Variable | Default | Required |
|----------|---------|----------|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | No |
| `MODEL_NAME` | `Qwen/Qwen2.5-3B-Instruct` | No |
| `API_KEY` | - | Yes |

### Output Format

The script follows the hackathon output format:

```
[START] task=wildfire_detection env=wildfire_detection model=Qwen/Qwen2.5-3B-Instruct
[STEP] step=1 action=Alert reward=-1.00 done=false error=null
[STEP] step=2 action=Scan reward=0.50 done=false error=null
[END] success=true steps=16 rewards=-1.00,0.50,2.00,...
```

## Actions

| Action | Description |
|--------|-------------|
| `Alert` | Raise immediate alert if fire/smoke detected |
| `Scan` | Continue scanning/observing the current frame |
| `Ignore` | No action needed, clear conditions |
| `Deploy` | Deploy resources to the detected location |

## Reward Structure

| Reward | Condition |
|--------|-----------|
| +2.00 | Correct detection with confidence ≥ 0.85 and clear Grad-CAM alignment |
| +1.00 | Correct class but lower confidence |
| +0.50 | Partial credit (right class, weak XAI) |
| -0.75 | False positive (Alert/Deploy on no_fire) |
| -1.00 | False negative (missed fire/smoke) |
| -0.50 | Inaction ("Ignore") when fire/smoke is present |
| -2.00 | Terminal failure (too many missed detections) |

## Grading

The grader evaluates each episode:
- **Success**: All fire/smoke frames correctly alerted with zero false positives
- **Score**: Normalized 0.0-1.0 based on cumulative reward

## Project Structure

```
multipen/
├── inference.py                      # LLM agent inference script
├── wildfire_env.py                   # RL environment (Gym-compatible)
├── firenet_inference.py              # FirenetCNN + Grad-CAM
├── models.py                         # Action/Observation models
├── openenv.yaml                      # OpenEnv configuration
├── pyproject.toml                    # Dependencies
├── Dockerfile                        # Container definition
├── server/
│   ├── app.py                        # FastAPI server
│   └── wildfire_environment.py       # OpenEnv wrapper
└── environments/
    └── wildfire_detection/
        ├── wildfire_env.py           # Environment implementation
        ├── firenet_inference.py      # Model inference
        └── sample_frames/            # Test images (16 frames)
```

## Deployment to Hugging Face Spaces

```bash
# Push to Hugging Face
openenv push

# Or with options
openenv push --namespace my-org --private
```

## Testing

```bash
# Test environment directly
python test_env.py

# Test API
curl http://localhost:8000/health
curl http://localhost:8000/schema
```

## Model

Uses FirenetCNN (MobileNetV2-based) from:
https://github.com/OpelSpeedster/Forest-Fire-Detection-Using-FirenetCNN-and-XAI-Techniques

The model classifies images into:
- `no_fire` - No fire detected
- `fire` - Fire detected
- `smoke` - Smoke detected