import requests
import argparse
import subprocess

API_URL = "http://localhost:11434/v1/chat/completions"

SYSTEM_PROMPT = """你是一个经验丰富的 Shell 大师，擅长 Linux 命令行操作。你的任务是将用户的自然语言需求转换为安全、准确的 Shell 命令。
- 确保命令高效且适用于大多数 Linux 发行版。
- 避免危险命令（如 `rm -rf /`）。
- 仅返回 Shell 命令，不要包含多余解释。
"""


def generate_command(natural_text):
    payload = {
        "model": "llama3-CN",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"请将以下自然语言转换为 Shell 命令：\n\n'{natural_text}'"}
        ]
    }
    response = requests.post(API_URL, json=payload)
    response_data = response.json()
    return response_data["choices"][0]["message"]["content"].strip()


def main():
    parser = argparse.ArgumentParser(description="自然语言转 Shell 命令工具")
    parser.add_argument("query", type=str, help="要转换的自然语言描述")
    parser.add_argument("--run", action="store_true", help="是否直接执行生成的命令")
    args = parser.parse_args()

    command = generate_command(args.query).replace("`", "")
    print(f"生成的命令: {command}")

    if args.run:
        confirm = input("是否执行该命令？(y/N): ").strip().lower()
        if confirm == "y":
            subprocess.run(command, shell=True)


if __name__ == "__main__":
    main()
