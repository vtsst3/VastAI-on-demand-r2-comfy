import asyncio
import json
import uuid
import uvicorn
import websockets
import boto3
import os
import base64
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from botocore.exceptions import NoCredentialsError, ClientError
from typing import Dict, Any

# --- 設定 ---
COMFYUI_URL = "127.0.0.1:8188"
# WORKFLOW_TEMPLATE_PATHは起動スクリプトがcloneするディレクトリを指すように修正
WORKFLOW_TEMPLATE_PATH = "/workspace/VastAI-on-demand-r2-comfy/workflow_template.json"
# LORA_LOCAL_DIRはlocal volume上のパスを想定
LORA_LOCAL_DIR = "/workspace/ComfyUI/models/loras"

# --- Cloudflare R2 設定 ---
R2_ENDPOINT_URL = "https://c89a44932ae8cbdc49a8a3d25830ecd9.r2.cloudflarestorage.com"
R2_BUCKET_NAME = "my-comfyui-models-2025"


# --- FastAPIアプリケーション ---
app = FastAPI()

# --- CORSミドルウェアの設定 ---
# フロントエンドからのリクエストを許可するために必要
origins = [
    "http://localhost:5174",  # フロントエンドのオリジン
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- グローバル変数 ---
workflow_template = {}
r2_client = None # s3_clientからr2_clientに名称変更
jobs: Dict[str, Dict[str, Any]] = {} # ジョブの状態と結果を保存するストア

# --- ヘルパー関数 (WebSocket部分は変更なし) ---
async def get_images(prompt_id, server_address):
    """WebSocket経由で画像バイナリデータを直接受信する"""
    uri = f"ws://{server_address}/ws?clientId={uuid.uuid4().hex}"
    try:
        # タイムアウトを300秒（5分）に設定
        async with websockets.connect(uri, ping_interval=20, ping_timeout=30, max_size=15 * 1024 * 1024) as websocket:
            while True:
                # タイムアウトを設定してメッセージを待つ
                out = await asyncio.wait_for(websocket.recv(), timeout=300.0)
                if isinstance(out, bytes):
                    return out  # 画像のバイナリデータを返す
    except asyncio.TimeoutError:
        print(f"WebSocket receive timed out for prompt_id: {prompt_id}")
        return None
    except Exception as e:
        print(f"WebSocket connection error for prompt_id {prompt_id}: {e}")
        return None

def queue_prompt(prompt_workflow, server_address):
    import requests
    p = {"prompt": prompt_workflow}
    data = json.dumps(p).encode('utf-8')
    req = requests.post(f"http://{server_address}/prompt", data=data)
    req.raise_for_status()
    return req.json()

async def download_lora_from_r2(r2_path: str):
    """Cloudflare R2からLoRAファイルをダウンロードする"""
    if not r2_client:
        raise Exception("R2クライアントが初期化されていません。")
    try:
        filename = os.path.basename(r2_path)
        local_path = os.path.join(LORA_LOCAL_DIR, filename)
        if not os.path.exists(local_path):
            print(f"R2オブジェクト '{r2_path}' を '{local_path}' へダウンロード中...")
            os.makedirs(LORA_LOCAL_DIR, exist_ok=True)
            r2_client.download_file(R2_BUCKET_NAME, r2_path, local_path)
            print("ダウンロード完了。")
        else:
            print(f"LoRAファイル '{filename}' は既に存在します。ダウンロードをスキップします。")
        return filename
    except NoCredentialsError:
        raise Exception("Cloudflare R2の認証情報(AWS互換)が見つかりません。")
    except ClientError as e:
        raise Exception(f"R2からのダウンロード中にエラーが発生しました: {e}")
    except Exception as e:
        raise Exception(f"LoRAのダウンロード処理中に予期せぬエラーが発生しました: {e}")


# --- バックグラウンドタスク ---
async def run_generation_task(job_id: str, req_data: dict):
    """画像生成処理をバックグラウンドで実行する"""
    global jobs
    try:
        jobs[job_id]["status"] = "processing"
        
        model_type = req_data.get("model_type")
        positive_prompt = req_data.get("positive_prompt", "")
        negative_prompt = req_data.get("negative_prompt", "")
        seed = req_data.get("seed", -1)
        # lora_s3_pathキーはそのまま流用し、値にはR2のオブジェクトキー(例: "loras/my_lora.safetensors")を期待する
        lora_r2_path = req_data.get("lora_s3_path")
        lora_strength = req_data.get("lora_strength", 1.0)

        prompt_workflow = json.loads(json.dumps(workflow_template))

        if lora_r2_path:
            lora_filename = await download_lora_from_r2(lora_r2_path)
            for node_id in ["121", "179"]:
                if node_id in prompt_workflow:
                    prompt_workflow[node_id]["inputs"]["lora_name"] = lora_filename
                    prompt_workflow[node_id]["inputs"]["strength_model"] = lora_strength
                    prompt_workflow[node_id]["inputs"]["strength_clip"] = lora_strength
        
        for node_id, node_info in prompt_workflow.items():
            if node_info["class_type"] == "Seed (rgthree)":
                node_info["inputs"]["seed"] = seed if seed != -1 else node_info["inputs"]["seed"]

        switch_nodes = ["192", "193", "195"]
        if model_type == "sdxl":
            prompt_workflow["80"]["inputs"]["text"] = positive_prompt
            prompt_workflow["76"]["inputs"]["text"] = negative_prompt
            for node_id in switch_nodes:
                if node_id in prompt_workflow: prompt_workflow[node_id]["inputs"]["select"] = 1
        elif model_type == "qwen_sdxl":
            prompt_workflow["182"]["inputs"]["text"] = positive_prompt
            prompt_workflow["183"]["inputs"]["text"] = negative_prompt
            for node_id in switch_nodes:
                if node_id in prompt_workflow: prompt_workflow[node_id]["inputs"]["select"] = 2

        queued_job = queue_prompt(prompt_workflow, COMFYUI_URL)
        prompt_id = queued_job['prompt_id']
        
        image_binary = await get_images(prompt_id, COMFYUI_URL)

        if not image_binary:
            raise Exception("Failed to receive image data from ComfyUI via WebSocket.")

        image_data = image_binary[8:]
        encoded_string = base64.b64encode(image_data).decode('utf-8')
        
        # ジョブストアに結果を保存
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = {
            "message": "処理が完了しました。",
            "image_base64": encoded_string
        }
    except Exception as e:
        print(f"バックグラウンドタスクでエラー発生 (Job ID: {job_id}): {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


# --- APIエンドポイント ---
@app.on_event("startup")
async def startup_event():
    """アプリケーション起動時の初期化処理"""
    global workflow_template, r2_client
    try:
        with open(WORKFLOW_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
            workflow_template = json.load(f)
        print(f"'{WORKFLOW_TEMPLATE_PATH}' からワークフローテンプレートを正常に読み込みました。")
    except Exception as e:
        print(f"致命的エラー: ワークフローファイル '{WORKFLOW_TEMPLATE_PATH}' の読み込みに失敗しました: {e}")
        workflow_template = None

    # Cloudflare R2クライアントの初期化
    try:
        if not all([os.environ.get('AWS_ACCESS_KEY_ID'), os.environ.get('AWS_SECRET_ACCESS_KEY')]):
            raise Exception("環境変数 AWS_ACCESS_KEY_ID と AWS_SECRET_ACCESS_KEY が設定されていません。")

        r2_client = boto3.client(
            's3', # R2はS3互換APIなので 's3' を指定
            endpoint_url=R2_ENDPOINT_URL,
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            region_name='auto' # R2では 'auto' が推奨される
        )
        print("Cloudflare R2クライアントを正常に初期化しました。")
    except Exception as e:
        print(f"警告: R2クライアントの初期化に失敗しました: {e}")
        r2_client = None

@app.post("/invoke")
async def invoke_background(request: Request, background_tasks: BackgroundTasks):
    """画像生成ジョブを受け付け、バックグラウンドで実行する"""
    if not workflow_template:
        raise HTTPException(status_code=500, detail="サーバーエラー: ワークフローテンプレートが読み込まれていません。")

    req_data = await request.json()
    model_type = req_data.get("model_type")

    if model_type not in ["sdxl", "qwen_sdxl"]:
        raise HTTPException(status_code=400, detail="model_typeは 'sdxl' または 'qwen_sdxl' である必要があります。")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending"}
    
    background_tasks.add_task(run_generation_task, job_id, req_data)
    
    return JSONResponse(status_code=202, content={"job_id": job_id, "status": "processing"})

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """指定されたジョブIDのステータスを返す"""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(status_code=200, content={"job_id": job_id, "status": job.get("status"), "error": job.get("error")})

@app.get("/result/{job_id}")
async def get_result(job_id: str):
    """指定されたジョブIDの結果を返す"""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("status") != "completed":
        raise HTTPException(status_code=400, detail=f"Job is not completed. Current status: {job.get('status')}")
    
    result = job.get("result")
    # 結果を返したらメモリから削除
    del jobs[job_id]
    
    return JSONResponse(status_code=200, content=result)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
