
import argparse
from huggingface_hub import snapshot_download
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_repo', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--token')
    
    args = parser.parse_args()
    
    try:
        os.makedirs(args.save_path, exist_ok=True)
        snapshot_download(
            repo_id=args.model_repo,
            local_dir=args.save_path,
            token=args.token,
            local_dir_use_symlinks=False
        )
        print(f"✅ 模型已成功下载至 {args.save_path}")
    except Exception as e:
        print(f"❌ 下载失败: {str(e)}")

if __name__ == "__main__":
    main()